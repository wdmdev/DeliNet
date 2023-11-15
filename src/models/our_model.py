import numpy as np
import torch
from torch import nn
import torch.nn.functional as f
import tqdm
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from torchvision import transforms
import os
from torch.utils.data import DataLoader
from src.data.our_kaggle_food_dataset import KaggleFoodDataset
from src.evaluation.get_top_x_acc import get_top_x_acc
from src.models.model_utils import *


def train_our_model(csv_file_path, image_dir, vision_model, text_model, batch_size=40, lr=0.0001, d="cuda"):
    vision_model = vision_model.to(d)
    text_model = text_model.to(d)

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


    dataloader_train = DataLoader(food_dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(food_dataset_test, batch_size=batch_size, shuffle=False)


    loss_fn = torch.nn.CrossEntropyLoss()
    lr = lr * (batch_size / 100) # 0.00003 for Vit (since smaller batchsize) 0.0001 for resnet
    opt_vision = torch.optim.AdamW(lr=lr, params=vision_model.parameters())
    opt_text = torch.optim.AdamW(lr=lr, params=text_model.parameters())

    losses = []
    top_x_acc_list = []
    num_steps = []

    for epoch_num in range(100):
        vision_model.train()
        text_model.train()
        for batch_num, (images, titles, ingres, decs) in enumerate(tqdm.tqdm(dataloader_train)):
            labels = torch.arange(images.shape[0], device=d)
            images = images.to(d)

            text_latent = vision_model(images)
            img_latent = text_model(titles, ingres, decs)
            text_latent = text_latent / torch.linalg.norm(text_latent, axis=1, keepdim=True)
            img_latent = img_latent / torch.linalg.norm(img_latent, axis=1, keepdim=True)

            logits = (text_latent @ img_latent.T) * torch.exp(text_model.t)

            loss_i = loss_fn(logits, labels)
            loss_t = loss_fn(logits.T, labels)
            loss = (loss_i + loss_t) / 2.0

            opt_vision.zero_grad()
            opt_text.zero_grad()

            loss.backward()

            opt_vision.step()
            opt_text.step()

            losses.append(loss.detach())


        top_x_percent = [0, 0.10, 0.25]
        top_x_acc = get_top_x_acc(logits=None, top_x_percent = top_x_percent, test_loader=dataloader_test,
                                  text_model=text_model, vision_model=vision_model, d=d)
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
        plt.title(f"{text_model.__class__.__name__} & {vision_model.model.name_or_path} - epoch {epoch_num+1} ")
        plt.xlabel("steps")
        plt.ylabel("acc/ normalized loss")
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    food_dir = os.path.join(current_dir,'..','..','data','processed','KaggleFoodDataset')
    csv_file_path = os.path.join(food_dir, 'data.csv')
    image_dir = os.path.join(food_dir,'images')


    from transformers import BertTokenizer, BertModel, ViTModel, ViTMAEModel, AutoImageProcessor
    # ViTImageProcessor_ = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")# uses same preprocess as resnet!

    #ViTModel_ = ViTModel.from_pretrained("facebook/vit-mae-base")
    #ViTModel_ = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
    #vision_model = ViT_wrapper(ViTModel_, d=d).to(d)
    # #vision_model(batch_images)

    BertModel_ = BertModel.from_pretrained("bert-base-uncased")
    BertTokenizer_ = BertTokenizer.from_pretrained('bert-base-uncased')
    #text_model = Bert_mono_wrapper(BertModel_, BertTokenizer_, d=d).to(d)
    #text_model = Bert_2xinput_wrapper(BertModel_, BertTokenizer_)
    #text_model = Bert_2x_network_wrapper(BertModel_, BertTokenizer_, d=d).to(d)

    from transformers import ResNetModel
    ResNet = ResNetModel.from_pretrained("microsoft/resnet-50")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=2, type=int)

    args = parser.parse_args()

    print(args.batch_size)
    d = "cpu"
    vision_model = ResNet_wrapper(ResNet, d=d)
    text_model = Bert_2xinput_wrapper(BertModel_, BertTokenizer_, d=d)
    batch_size = 2
    train_our_model(csv_file_path,
                    image_dir,
                    vision_model,
                    text_model,
                    batch_size=batch_size,
                    lr=0.0001,
                    d=d)







