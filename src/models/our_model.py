import numpy as np
import torch
from torch import nn
import torch.nn.functional as f
import tqdm
import time
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from torchvision import transforms
import os
import json
from torch.utils.data import DataLoader
from src.data.our_kaggle_food_dataset import KaggleFoodDataset
from src.evaluation.get_top_x_acc import get_top_x_acc
from src.models.model_utils import *
from loss_funcs import triplet_loss, contrastive_loss
from itertools import chain

def train_our_model(csv_file_path, image_dir, vision_model, text_model, loss_fn = contrastive_loss,
                    batch_size=40, lr=0.0001, d="cuda", num_epochs = 100, max_time = 10_000,
                    training_loop_test=False, save_results=False, use_mixed_precision=True):
    vision_model = vision_model.to(d) # model always starts on CPU and is then moved if needed
    vision_model.d = d

    text_model = text_model.to(d) # model always starts on CPU and is then moved if needed
    text_model.d = d

    vision_model_name = vision_model.__class__.__name__[:-8]+vision_model.size
    text_model_name = text_model.__class__.__name__[:-8]
    combined_name = vision_model_name + "_AND_" + text_model_name

    if save_results:
        save_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models", combined_name)
        os.makedirs(save_folder, exist_ok=True)

    if training_loop_test:
        print("Not training! Running in training_loop_test=True mode")

    vision_model = vision_model.to(d)
    text_model = text_model.to(d)
    vision_model_nparams = int(sum([param.numel() for param in vision_model.parameters()])/ 1e6)
    text_model_nparams = int(sum([param.numel() for param in text_model.parameters()]) / 1e6)
    print(f"{vision_model_name}: {vision_model_nparams}m params")
    print(f"{text_model_name}: {text_model_nparams}m params")
    #print(sum([param.numel() for param in vision_model.parameters()])/1e6)

    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    #batch_size = 2 #200 is max for resnet18, 80 is max for resnet50, 44 is max for ViT
    food_dataset_train = KaggleFoodDataset(csv_file=csv_file_path, image_dir=image_dir,
                                           transform=preprocess, train=True, train_split=0.9)
    food_dataset_test = KaggleFoodDataset(csv_file=csv_file_path, image_dir=image_dir,
                                           transform=preprocess, train=False, train_split=0.9)

    num_workers = 0 if os.name=="nt" else 6 #os.cpu_count()# set num workers to all cores if not windows
    dataloader_train = DataLoader(food_dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_test = DataLoader(food_dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    #loss_fn = torch.nn.CrossEntropyLoss()
    lr = lr * (batch_size / 100) # 0.00003 for Vit (since smaller batchsize) 0.0001 for resnet
    combined_params = chain(vision_model.parameters(), text_model.parameters())
    opt = torch.optim.AdamW(lr=lr, params=combined_params)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.01, total_iters=50)
    AMP_scaler = torch.cuda.amp.GradScaler()

    losses = []
    best_25_perf = 0.0
    top_x_percent = [0, 0.10, 0.25]
    top_x_acc_list = [top_x_percent.copy()]
    num_steps = [0]

    start_time = time.time()
    for epoch_num in range(num_epochs):
        if time.time() - start_time > max_time:
            print(f"time limit of {max_time:.0f}s reached!")
            break

        vision_model.train()
        text_model.train()
        for batch_num, (images, text) in enumerate(tqdm.tqdm(dataloader_train, unit="batch", desc=f"Epoch {epoch_num}")):
            labels = torch.arange(images.shape[0], device=d)
            images = images.to(d)

            # mixed precision
            with torch.amp.autocast(enabled=use_mixed_precision, device_type=d, dtype=torch.float16):
                text_latent = vision_model(images)
                img_latent = text_model(text)

                text_latent = text_latent / torch.linalg.norm(text_latent, axis=1, keepdim=True)
                img_latent = img_latent / torch.linalg.norm(img_latent, axis=1, keepdim=True)

                loss = loss_fn(text_latent, img_latent, labels, text_model)

            opt.zero_grad()
            if use_mixed_precision:
                AMP_scaler.scale(loss).backward()
                AMP_scaler.step(opt)
                AMP_scaler.update()
            else:
                loss.backward()
                opt.step()


            losses.append(loss.detach())

            if training_loop_test and batch_num == 2: break

        lr_scheduler.step()

        top_x_acc = get_top_x_acc(logits=None, top_x_percent = top_x_percent, test_loader=dataloader_test,
                                  text_model=text_model, vision_model=vision_model, d=d,
                                  test_training_loop=training_loop_test)
        top_x_acc_list.append(top_x_acc)
        print("top x test acc", *zip(top_x_percent, top_x_acc))
        losses = losses[:-1] # final loss is kinda broken due to smaller batchsize
        num_steps.append(len(losses)-1)

        top_x_acc_array = np.asarray(top_x_acc_list).T
        for i in range(len(top_x_percent)):
            plt.plot(num_steps, top_x_acc_array[i],
                     label=f"top {int(top_x_percent[i]*100)}% acc = {top_x_acc_array[i][-1]:.3f}")
        losses_numpy = torch.stack(losses).cpu().numpy()
        plt.plot(losses_numpy/losses_numpy.max(), label="train loss (normalized)")
        plt.legend()
        plt.title(f"(epoch {epoch_num+1} batch size {batch_size}) \n {vision_model_name} & {text_model_name}")
        plt.xlabel("steps")
        plt.ylabel("acc/ normalized loss")
        plt.tight_layout()

        if save_results:
            data_dict = {"batch_size": batch_size,
                         "lr": lr,
                         "epochs": epoch_num,
                         "loss": losses_numpy.tolist(),
                         "top x test acc:": top_x_acc_list,
                         "num_steps": num_steps}

            json.dump(data_dict, open(os.path.join(save_folder, "meta.json"), 'w'))
            plt.savefig(os.path.join(save_folder, "training_plot.png"), dpi=200)

        if save_results and best_25_perf < top_x_acc_list[-1][-1]:
            best_25_perf = top_x_acc_list[-1][-1]
            torch.save(vision_model, os.path.join(save_folder, vision_model_name+".pt"))
            torch.save(text_model, os.path.join(save_folder, text_model_name+".pt"))

            # can be loaded just as:
            # vision_model = torch.load(os.path.join(save_folder, vision_model_name+".pt"))
            # text_model = torch.load(os.path.join(save_folder, text_model_name+".pt"))

        plt.show(block=False)
        plt.close('all')
        if training_loop_test:
            print(f"Peak memory usage: {torch.cuda.max_memory_allocated()/ 1e9:.1f}GB")
            print("Test complete - exiting test training!")
            break


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    food_dir = os.path.join(current_dir,'..','..','data','processed','KaggleFoodDataset')
    csv_file_path = os.path.join(food_dir, 'data.csv')
    image_dir = os.path.join(food_dir,'images')

    #torch.set_default_dtype(torch.float16)

    d = "cuda"
    vision_model = ResNet_wrapper(size = "18")
    #vision_model = ViT_wrapper()
    #vision_model = EfficientTrans_wrapper(d=d)
    #vision_model = Efficientnet_wrapper(size=3, d=d)
    #text_model = Bert_2x_network_wrapper(BertModel_, BertTokenizer_, d=d)
    #text_model = Bert_2xinput_wrapper(d=d)
    text_model = DistilBert_mono_wrapper()
    batch_size = 450
    training_loop_test = False
    save_results = True
    train_our_model(csv_file_path,
                    image_dir,
                    vision_model,
                    text_model,
                    batch_size=batch_size,
                    lr=0.0001,
                    d=d,
                    num_epochs = 200,
                    max_time = 3600*2,  #1hour
                    use_mixed_precision=True,
                    training_loop_test=training_loop_test,
                    save_results=save_results)







