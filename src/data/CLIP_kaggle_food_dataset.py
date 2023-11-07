import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import clip

import hydra
from omegaconf import DictConfig

class KaggleFoodDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=transforms.ToTensor()):
        """
        Args:
            csv_file (string): Path to the csv file with recipe and image mappings.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on the image.
        """
        food_data = pd.read_csv(csv_file)
        self.recipe_titles = clip.tokenize(list(food_data.iloc[:, 1].astype(str)))
        self.img_paths = [os.path.join(image_dir, img_name + ".jpg") for img_name in food_data.iloc[:, 4]]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx])

        if self.transform:
            image = self.transform(image)

        recipe_title = self.recipe_titles[idx] 

        return image, recipe_title


@hydra.main(version_base=None, config_path=os.path.join('..', '..', 'conf'), config_name='config')
def test(cfg: DictConfig):
    # Create the dataset and dataloader using the new class
    current_dir = os.path.dirname(os.path.abspath(__file__))
    food_dir = os.path.join(current_dir,'..','..','data','processed','KaggleFoodDataset')
    csv_file_path = os.path.join(food_dir, 'data.csv')
    image_dir = os.path.join(food_dir,'images')

    device = "cuda" if torch.cuda.is_available() else "cpu" 
    _, preprocess = clip.load(cfg.model.load, device=device, jit=False)

    def my_transform(img):
        img = img.convert("RGB")
        img = transforms.Resize((224,224))(img)
        img = preprocess(img)

        return img

    batch_size = 4
    food_dataset_batched = KaggleFoodDataset(csv_file=csv_file_path, image_dir=image_dir, transform=my_transform)
    food_dataloader_batched = DataLoader(food_dataset_batched, batch_size=batch_size, shuffle=True)

    # Test the DataLoader by fetching one batch of data
    batch_images, batch_recipe_titles = next(iter(food_dataloader_batched))

    # Show shapes for verification
    print(f'Batch size: {batch_size}')
    print(f'Image batch shape: {batch_images.shape}')
    print(f'Type of image batch: {type(batch_images)}')
    print(f'Recipe titles: {batch_recipe_titles}')
    print(f'Type of recipe titles: {type(batch_recipe_titles)}')
    print(f'Type of recipe titles element: {type(batch_recipe_titles[0])}')


if __name__ == '__main__':
    test()