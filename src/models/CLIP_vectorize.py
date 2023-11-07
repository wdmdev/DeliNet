import os
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    #Load the state_dict from the checkpoint
    checkpoint = torch.load("models/CLIP1/epoch=29-step=1350.ckpt")
    state_dict = checkpoint['state_dict']

    # Load the model from Hugging Face and apply the state_dict
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", state_dict=state_dict)
    model = model.to(device)
    model.eval()

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    #Load images from data/processed/images
    image_dir = "data/processed/KaggleFoodDataset/images"
    image_files = os.listdir(image_dir)


    inputs = {}

    for img in tqdm(image_files):
        with Image.open(os.path.join(image_dir, img)) as image:
            processed = processor(text=[""], images=image, return_tensors='pt', padding=True)
            processed = {name: tensor.to(device) for name, tensor in processed.items()}
    
            if not inputs:
                inputs = processed
            else:
                inputs = {name: torch.cat((inputs[name], processed[name]), dim=0) for name in processed}

    image_embeds, _ = model(**inputs)

    # Assuming 'outputs' is the result from your CLIP model
    image_embeddings = image_embeds.cpu().detach().numpy()

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=30, random_state=0).fit(image_embeddings)

    # Find the closest image to each cluster center
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Create directory for cluster images if it doesn't exist
    cluster_dir = 'cluster_img'
    os.makedirs(cluster_dir, exist_ok=True)

    # Identify and save the closest images to the cluster centers
    for i, center in enumerate(cluster_centers):
        # Calculate distances of embeddings within the same cluster to the center
        within_cluster_indices = np.where(cluster_labels == i)[0]
        within_cluster_embeddings = image_embeddings[within_cluster_indices]
        distances = np.linalg.norm(within_cluster_embeddings - center, axis=1)
        closest_image_idx = within_cluster_indices[np.argmin(distances)]

        # Retrieve the image data for the closest image
        # Here, you would retrieve the actual image using whatever method you have to map indices to images
        # For the purpose of this example, we will assume a function get_image_by_index that retrieves the image:
        # closest_image_data = get_image_by_index(closest_image_idx)

        # Save the closest image
        image_path = os.path.join(cluster_dir, f'cluster_{i}.png')
        # closest_image_data.save(image_path)

    # Note: The function `get_image_by_index` needs to be defined by the user
    # It should return the image in PIL Image format for the given index