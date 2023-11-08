import torch
from torch import nn
import torch.nn.functional as f


class ViT_wrapper(torch.nn.Module):
    def __init__(self, ViTModel, ViTImageProcessor, latent_dim=768, d="cuda"):
        super().__init__()
        self.d = d
        self.latent_dim = latent_dim
        self.ViTModel = ViTModel
        self.ViTImageProcessor = ViTImageProcessor
        self.fc = torch.nn.Linear(latent_dim, latent_dim)

    def forward(self, images):
        preprocessed = self.ViTImageProcessor(images=images, return_tensors="pt").to(self.d)
        output = self.ViTModel(**preprocessed)
        cls_token = output.pooler_output
        out = self.fc(cls_token)

        return out

class Bert_wrapper(torch.nn.Module):
    def __init__(self, BertModel, BertTokenizer, latent_dim=768, d="cuda", max_length=16):
        super().__init__()
        self.d = d
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.BertModel = BertModel
        self.BertTokenizer = BertTokenizer
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
        out = self.fc(cls_token)

        return out



if __name__ == "__main__":
    # if torch.cuda.is_available():
    #     torch.set_default_device("cuda")
    # else:
    #     torch.set_default_device("cpu")
    d = "cpu"

    from torch.utils.data import DataLoader
    from src.data.kaggle_food_dataset import *
    current_dir = os.path.dirname(os.path.abspath(__file__))
    food_dir = os.path.join(current_dir,'..','..','data','processed','KaggleFoodDataset')
    csv_file_path = os.path.join(food_dir, 'data.csv')
    image_dir = os.path.join(food_dir,'images')

    batch_size = 3
    food_dataset_batched = KaggleFoodDataset(csv_file=csv_file_path, image_dir=image_dir)
    food_dataloader_batched = DataLoader(food_dataset_batched, batch_size=batch_size, shuffle=True)

    batch_images, batch_recipe_titles = next(iter(food_dataloader_batched))


    from transformers import BertTokenizer, BertModel, AutoImageProcessor, ViTMAEModel, ViTImageProcessor, ViTModel


    #ViTImageProcessor_ = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    #ViTModel_ = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    #vision_model = ViT_wrapper(ViTModel_, ViTImageProcessor_, d=d).to(d)
    ViTImageProcessor_ = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    ViTModel_ = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
    vision_model = ViT_wrapper(ViTModel_, ViTImageProcessor_, d=d).to(d)
    #vision_model(batch_images)


    BertModel_ = BertModel.from_pretrained("bert-base-uncased")
    BertTokenizer_ = BertTokenizer.from_pretrained('bert-base-uncased')
    text_model = Bert_wrapper(BertModel_, BertTokenizer_, d=d).to(d)
    #batch_recipe_titles = list(batch_recipe_titles[0])
    #text_model(batch_recipe_titles)


    print(torch.empty(3, dtype=torch.long).random_(5))
    import tqdm

    labels = torch.arange(batch_size).to(d)
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 1e-4
    opt_vision = torch.optim.AdamW(lr=lr, params=vision_model.parameters())
    opt_text = torch.optim.AdamW(lr=lr, params=text_model.parameters())

    for batch_images, batch_recipe_titles in tqdm.tqdm(food_dataloader_batched):

        #batch_recipe_titles = list(batch_recipe_titles[0])

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




