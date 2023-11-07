import torch
from torch import nn
from model_utils import DoubleConv, Down, SelfAttention, pos_encoding
import torch.nn.functional as f

class Image_encoder_baseline(torch.nn.Module):
    def __init__(self, img_size=16, c_in=3, latent_dim=5, time_dim=256, channels=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, channels)
        self.down1 = Down(channels, channels * 2, emb_dim=time_dim)
        self.sa1 = SelfAttention(channels * 2, img_size // 2)
        self.down2 = Down(channels * 2, channels * 4, emb_dim=time_dim)

        self.sa2 = SelfAttention(channels * 4, img_size // 4)
        self.down3 = Down(channels * 4, channels * 4, emb_dim=time_dim)
        self.sa3 = SelfAttention(channels * 4, img_size // 8)

        self.bot1 = DoubleConv(channels * 4, channels * 8)
        self.bot2 = DoubleConv(channels * 8, channels * 8)
        #self.bot3 = DoubleConv(channels * 8, channels * 4)
        self.MLP1 = nn.Linear(int((16 * 3 * 32 + self.time_dim * 2) / 2), int((16 * 3 * 32 + self.time_dim * 2) / 4))
        self.MLP2_out = nn.Linear(int((16 * 3 * 32 + self.time_dim * 2) / 4), self.latent_dim)

    def forward(self, x, t=None):
        # t will always be None
        #t = t.unsqueeze(-1).type(torch.float)
        #t = pos_encoding(t, self.time_dim, self.device)

        print(x.shape)
        x = self.inc(x)
        print(x.shape)
        x = self.down1(x, t)
        print(x.shape)
        x = self.sa1(x)
        print(x.shape)
        x = self.down2(x, t)
        print(x.shape)
        x = self.sa2(x)
        print(x.shape)
        x = self.down3(x, t)
        print(x.shape)
        x = self.sa3(x)
        print(x.shape)
        x = self.bot1(x)
        print(x.shape)
        x = self.bot2(x)
        print(x.shape)
        # x = self.bot3(x)
        x = x.flatten(start_dim=1)
        print(x.shape)
        x = f.relu(self.MLP1(x))
        print(x.shape)
        out = self.MLP2_out(x)
        print(out.shape)

        return out


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    else:
        torch.set_default_device("cpu")

    import sys
    import os
    #sys.path.append(r"../../")
    #sys.path.append(r"C:\Users\karl\Desktop\DeliNet")
    #for path in sys.path: print(path)
    from torch.utils.data import DataLoader
    #sys.path.append("..")

    #from ..data.kaggle_food_dataset import KaggleFoodDataset
    from src.data.kaggle_food_dataset import *
    #KaggleFoodDataset()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    food_dir = os.path.join(current_dir,'..','..','data','processed','KaggleFoodDataset')
    csv_file_path = os.path.join(food_dir, 'data.csv')
    image_dir = os.path.join(food_dir,'images')

    batch_size = 1
    food_dataset_batched = KaggleFoodDataset(csv_file=csv_file_path, image_dir=image_dir)
    food_dataloader_batched = DataLoader(food_dataset_batched, batch_size=batch_size, shuffle=True)

    # Test the DataLoader by fetching one batch of data
    batch_images, batch_recipe_titles = next(iter(food_dataloader_batched))

    img_size = int(128*2)
    t = torch.randn((batch_size, 3, img_size, img_size), dtype=torch.float)
    model = Image_encoder_baseline(img_size=img_size)
    with torch.no_grad():
        model(t)

    def some_func(some_int: int) -> float:
        return float(some_int)

    some_func(1.0)


    #iter(next(KaggleFoodDataset()))