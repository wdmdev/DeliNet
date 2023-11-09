import torch
from torch import nn
import torch.nn.functional as f
import tqdm
import matplotlib.pyplot as plt

class ResNet_wrapper(torch.nn.Module):
    def __init__(self, model, latent_dim=768, d="cuda"):
        super().__init__()
        self.d = d
        self.latent_dim = latent_dim
        self.model = model
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
    def __init__(self, ViTModel, ViTImageProcessor, latent_dim=768, d="cuda"):
        super().__init__()
        self.d = d
        self.latent_dim = latent_dim
        self.ViTModel = ViTModel
        self.ViTImageProcessor = ViTImageProcessor
        self.activation = torch.nn.GELU()
        self.fc = torch.nn.Linear(latent_dim, latent_dim)

    def forward(self, images):
        #preprocessed = self.ViTImageProcessor(images=images, return_tensors="pt", do_rescale=False).to(self.d)
        #output = self.ViTModel(**preprocessed)
        output = self.ViTModel(images)
        cls_token = output.pooler_output
        out = self.fc(self.activation(cls_token))

        return out

class Bert_wrapper(torch.nn.Module):
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

    def forward(self, text):
        preprocessed = self.BertTokenizer(text=text,
                                          padding=True,
                                          truncation=True,
                                          max_length=self.max_length,
                                          return_tensors="pt").to(self.d)

        output = self.BertModel(**preprocessed)
        cls_token = output.pooler_output
        out = self.fc(self.activation(cls_token))

        return out



if __name__ == "__main__":
    # if torch.cuda.is_available():
    #     torch.set_default_device("cuda")
    # else:
    #     torch.set_default_device("cpu")
    d = "cuda"

    from torch.utils.data import DataLoader
    from src.data.our_kaggle_food_dataset import *
    current_dir = os.path.dirname(os.path.abspath(__file__))
    food_dir = os.path.join(current_dir,'..','..','data','processed','KaggleFoodDataset')
    csv_file_path = os.path.join(food_dir, 'data.csv')
    image_dir = os.path.join(food_dir,'images')

    preprocess = transforms.Compose([
        #transforms.Resize((224,224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    batch_size = 44 #200 is max for resnet18, 80 is max for resnet50, 44 is max for ViT
    food_dataset_batched = KaggleFoodDataset(csv_file=csv_file_path, image_dir=image_dir, transform=preprocess)
    food_dataloader_batched = DataLoader(food_dataset_batched, batch_size=batch_size, shuffle=True)

    batch_images, batch_recipe_titles = next(iter(food_dataloader_batched))


    from transformers import BertTokenizer, BertModel, ViTModel, AutoImageProcessor

    # ViTImageProcessor_ = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")# uses same as resnet!
    ViTModel_ = ViTModel.from_pretrained("facebook/vit-mae-base")
    vision_model = ViT_wrapper(ViTModel_, None, d=d).to(d)
    # #vision_model(batch_images)

    BertModel_ = BertModel.from_pretrained("bert-base-uncased")
    BertTokenizer_ = BertTokenizer.from_pretrained('bert-base-uncased')
    text_model = Bert_wrapper(BertModel_, BertTokenizer_, d=d).to(d)

    # from transformers import ResNetModel
    # ResNet = ResNetModel.from_pretrained("microsoft/resnet-50")
    # vision_model = ResNet_wrapper(ResNet, d=d).to(d)

    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.00003 # 0.00003 for Vit (since smaller batchsize) 0.0001 for resnet
    opt_vision = torch.optim.AdamW(lr=lr, params=vision_model.parameters())
    opt_text = torch.optim.AdamW(lr=lr, params=text_model.parameters())

    losses = []

    for epoch_num in range(100):
        for batch_num, (batch_images, batch_recipe_titles) in enumerate(tqdm.tqdm(food_dataloader_batched)):
            labels = torch.arange(batch_images.shape[0], device=d)
            batch_images = batch_images.to(torch.float32).to(d)

            text_latent = vision_model(batch_images)
            img_latent = text_model(batch_recipe_titles)
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

            if batch_num == 0:
                print(f"\nacc of batch 0 of epoch {epoch_num}: "
                      f"{((torch.argmax(logits, dim=0) == labels).sum() / batch_size).item():.3f}"
                      f" (random is {1/batch_size:.3f})")

        losses = losses[:-1] # final loss is kinda broken due to smaller batchsize

        plt.plot(torch.stack(losses).cpu().numpy())
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




