import os
from PIL import Image
from tqdm import tqdm

def find_single_channel_images(folder_path):
    single_channel_images = []

    # Iterate over all files in the specified folder
    for filename in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        
        with Image.open(file_path) as img:
            # Check if the image has only one channel (e.g., grayscale)
            if img.mode == 'L':
                single_channel_images.append(filename)

    return single_channel_images

# Specify the folder path here
folder_path = '../../data/processed/KaggleFoodDataset/images'
single_channel_images = find_single_channel_images(folder_path)

# Print out the list of single channel images
print("Single channel images found:")
for image in single_channel_images:
    print(image)
