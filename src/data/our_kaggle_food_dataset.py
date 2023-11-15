import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class KaggleFoodDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, train=True, train_split=0.8):
        """
        Args:
            csv_file (string): Path to the csv file with recipe and image mappings.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on the image.
        """
        self.food_data = pd.read_csv(csv_file).dropna()
        if train:
            self.food_data = self.food_data.iloc[:int(len(self.food_data) * train_split), ...]
        else:
            self.food_data = self.food_data.iloc[int(len(self.food_data) * train_split):, ...]

        self.image_dir = image_dir
        self.titles = self.food_data.iloc[:,1].tolist()
        self.ingre = self.food_data.iloc[:,2].tolist()
        self.desc = self.food_data.iloc[:,3].tolist()
        self.img = self.food_data.iloc[:,4].tolist()
        self.transform = transform
        #self.all_data = torch.load(os.path.join(image_dir, "all_data_float16.pt"))


    def __len__(self):
        return len(self.food_data)

    # def __getitem__(self, idx):
    #     #img_name = os.path.join(self.image_dir, self.food_data.iloc[idx, 4] + ".jpg")
    #     #image = Image.open(img_name).convert('RGB')
    #     #image = self.transform(image)
    #     #image = transforms.ToTensor()(image)
    #     #image = torch.load(img_name).to(torch.float16)
    #     image = self.all_data[idx]
    #     recipe_title = self.food_data.iloc[idx, 1]
    #
    #     return image, recipe_title

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.img[idx] + ".jpg")
        image = Image.open(img_name).convert('RGB')
        # Resize images to ensure same dimensions, 224 because many models are pretrained on that
        #image = transforms.Resize((224,224))(image)

        image = self.transform(image)

        #torch.save(image, os.path.join(self.image_dir, self.food_data.iloc[idx, 4] + ".pt"))
        # recipe = {
        #     'title': [self.food_data.iloc[idx, 1]],
        #     'ingredients': [self.food_data.iloc[idx, 2]],
        #     'instructions': [self.food_data.iloc[idx, 3]]
        # }
        #image = transforms.ToTensor()(image).to(torch.float32)

        return image, self.titles[idx], self.ingre[idx], self.desc[idx]


def test():
    # Create the dataset and dataloader using the new class
    current_dir = os.path.dirname(os.path.abspath(__file__))
    food_dir = os.path.join(current_dir,'..','..','data','processed','KaggleFoodDataset')
    csv_file_path = os.path.join(food_dir, 'data.csv')
    image_dir = os.path.join(food_dir,'images')

    batch_size = 1000
    food_dataset_batched = KaggleFoodDataset(csv_file=csv_file_path, image_dir=image_dir)
    food_dataloader_batched = DataLoader(food_dataset_batched, batch_size=batch_size, shuffle=False)

    # Test the DataLoader by fetching one batch of data
    #batch_images, batch_recipe_titles = next(iter(food_dataloader_batched))
    import tqdm
    data = []
    for batch_images, _ in tqdm.tqdm(food_dataloader_batched):
        data.append(batch_images)

    # Show shapes for verification
    # print(f'Batch size: {batch_size}')
    # print(f'Image batch shape: {batch_images.shape}')
    # print(f'Type of image batch: {type(batch_images)}')
    # print(f'Recipe titles: {batch_recipe_titles}')
    # print(f'Type of recipe titles: {type(batch_recipe_titles)}')
    # print(f'Type of recipe titles element: {type(batch_recipe_titles[0])}')


if __name__ == '__main__':
    test()