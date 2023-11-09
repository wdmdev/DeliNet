import numpy as np
import torch
from torch import nn
import torch.nn.functional as f
import tqdm
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import copy
class ResNet_wrapper(torch.nn.Module):
    def __init__(self, ResNet, latent_dim=768, d="cuda"):
        super().__init__()
        self.d = d
        self.latent_dim = latent_dim
        self.model = ResNet
        self.activation = torch.nn.ReLU()

        if self.model.name_or_path.endswith(str(50)):
            self.fc = torch.nn.Linear(2048, latent_dim)

        elif self.model.name_or_path.endswith(str(18)):
            self.fc = torch.nn.Linear(512, latent_dim)

    def forward(self, images):
        output = self.model(images)
        output = output.pooler_output.squeeze()
        out = self.fc(self.activation(output))

        return out

class ViT_wrapper(torch.nn.Module):
    def __init__(self, ViT, latent_dim=768, d="cuda"):
        super().__init__()
        self.d = d
        self.latent_dim = latent_dim
        self.model = ViT
        self.activation = torch.nn.GELU()
        self.fc = torch.nn.Linear(latent_dim, latent_dim)

    def forward(self, images):
        output = self.model(images)
        #latent = output.latent
        latent = output.pooler_output
        out = self.fc(self.activation(latent))

        return out


class Bert_mono_wrapper(torch.nn.Module):
    def __init__(self, BertModel, BertTokenizer, latent_dim=768, d="cuda", max_length=16):
        super().__init__()
        self.d = d
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.BertModel = BertModel
        self.BertTokenizer = BertTokenizer
        self.activation = torch.nn.GELU()
        self.fc = torch.nn.Linear(latent_dim, latent_dim)
        self.t = torch.nn.Parameter(torch.tensor([0.07])) # init value from clip paper

    def forward(self, titles, ingredients=None):
        preprocessed = self.BertTokenizer(text=titles,
                                          padding=True,
                                          truncation=True,
                                          max_length=self.max_length,
                                          return_tensors="pt").to(self.d)

        output = self.BertModel(**preprocessed)
        cls_token = output.pooler_output
        out = self.fc(self.activation(cls_token))

        return out

class Bert_2xinput_wrapper(torch.nn.Module):
    def __init__(self, BertModel, BertTokenizer, latent_dim=768, d="cuda", max_length=128):
        super().__init__()
        self.d = d
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.BertModel = BertModel
        self.BertTokenizer = BertTokenizer
        self.activation = torch.nn.GELU()
        self.fc = torch.nn.Linear(latent_dim, latent_dim)
        self.t = torch.nn.Parameter(torch.tensor([0.07])) # init value from clip paper

    def forward(self, titles, ingredients):
        combined = [title + " " + ing for title, ing in zip(titles, ingredients)]
        preprocessed = self.BertTokenizer(text=combined,
                                          padding=True,
                                          truncation=True,
                                          max_length=self.max_length,
                                          return_tensors="pt").to(self.d)

        output = self.BertModel(**preprocessed)
        #output = self.BertModel(preprocessed.input_ids)
        cls_token = output.pooler_output
        out = self.fc(self.activation(cls_token))

        return out

class Bert_2x_network_wrapper(torch.nn.Module):
    def __init__(self, BertModel, BertTokenizer, latent_dim=768, d="cuda", max_length=32):
        super().__init__()
        self.d = d
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.BertModel1 = BertModel
        self.BertModel2 = copy.deepcopy(BertModel)

        self.BertTokenizer = BertTokenizer
        self.activation = torch.nn.GELU()
        self.fc1 = torch.nn.Linear(latent_dim*2, latent_dim*2)
        self.fc2 = torch.nn.Linear(latent_dim*2, latent_dim)

        self.t = torch.nn.Parameter(torch.tensor([0.07])) # init value from clip paper

    def forward(self, titles, ingredients):
        titles = self.BertTokenizer(text=titles,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_length,
                                    return_tensors="pt").to(self.d)

        ingredients = self.BertTokenizer(text=ingredients,
                                         padding=True,
                                         truncation=True,
                                         max_length=self.max_length*4,
                                         return_tensors="pt").to(self.d)
        #preprocessed["input_ids"]
        output1 = self.BertModel1(**titles).pooler_output
        output2 = self.BertModel2(**ingredients).pooler_output

        output = self.fc1(self.activation(torch.cat((output1, output2), dim=1)))
        out = self.fc2(self.activation(output))

        return out



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from src.data.our_kaggle_food_dataset import *
    from src.evaluation.get_top_x_acc import get_top_x_acc
    current_dir = os.path.dirname(os.path.abspath(__file__))
    food_dir = os.path.join(current_dir,'..','..','data','processed','KaggleFoodDataset')
    csv_file_path = os.path.join(food_dir, 'data.csv')
    image_dir = os.path.join(food_dir,'images')

    d = "cuda"
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    batch_size = 42 #200 is max for resnet18, 80 is max for resnet50, 44 is max for ViT
    food_dataset_train = KaggleFoodDataset(csv_file=csv_file_path, image_dir=image_dir,
                                           transform=preprocess, train=True, train_split=0.9)
    food_dataset_test = KaggleFoodDataset(csv_file=csv_file_path, image_dir=image_dir,
                                           transform=preprocess, train=False, train_split=0.9)


    dataloader_train = DataLoader(food_dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test= DataLoader(food_dataset_test, batch_size=batch_size, shuffle=False)

    from transformers import BertTokenizer, BertModel, ViTModel, ViTMAEModel, AutoImageProcessor
    # ViTImageProcessor_ = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")# uses same preprocess as resnet!

    #ViTModel_ = ViTModel.from_pretrained("facebook/vit-mae-base")
    #ViTModel_ = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
    #vision_model = ViT_wrapper(ViTModel_, d=d).to(d)
    # #vision_model(batch_images)

    BertModel_ = BertModel.from_pretrained("bert-base-uncased")
    BertTokenizer_ = BertTokenizer.from_pretrained('bert-base-uncased')
    #text_model = Bert_mono_wrapper(BertModel_, BertTokenizer_, d=d).to(d)
    text_model = Bert_2xinput_wrapper(BertModel_, BertTokenizer_, d=d).to(d)
    #text_model = Bert_2x_network_wrapper(BertModel_, BertTokenizer_, d=d).to(d)

    from transformers import ResNetModel
    ResNet = ResNetModel.from_pretrained("microsoft/resnet-50")
    vision_model = ResNet_wrapper(ResNet, d=d).to(d)

    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.0001 * (batch_size / 100) # 0.00003 for Vit (since smaller batchsize) 0.0001 for resnet
    opt_vision = torch.optim.AdamW(lr=lr, params=vision_model.parameters())
    opt_text = torch.optim.AdamW(lr=lr, params=text_model.parameters())

    losses = []
    top_x_acc_list = []
    num_steps = []

    for epoch_num in range(100):
        vision_model.train()
        text_model.train()
        for batch_num, (batch_images, batch_titles, batch_ingredients) in enumerate(tqdm.tqdm(dataloader_train)):
            labels = torch.arange(batch_images.shape[0], device=d)
            batch_images = batch_images.to(torch.float32).to(d)

            text_latent = vision_model(batch_images)
            img_latent = text_model(batch_titles, batch_ingredients)
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
    #print(loss.item())





    # I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
    # T_e = l2_normalize(np.dot(T_f, W_t), axis=1)
    # # scaled pairwise cosine similarities [n, n]
    # logits = np.dot(I_e, T_e.T) * np.exp(t)
    # # symmetric loss function
    # labels = np.arange(n)
    # loss_i = cross_entropy_loss(logits, labels, axis=0)
    # loss_t = cross_entropy_loss(logits, labels, axis=1)
    # loss = (loss_i + loss_t) / 2

    #
    # except Exception:
    #     print("#NAME?.jpg fuckery")




