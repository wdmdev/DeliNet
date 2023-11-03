import os

from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

import clip
from transformers import CLIPModel
from src.data.kaggle_food_dataset import KaggleFoodDataset

import hydra
from omegaconf import DictConfig, OmegaConf


# Function to convert model's parameters to FP32 format
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

@hydra.main(version_base=None, config_path=os.path.join('..', '..', 'conf'), config_name='config')
def finetune_model(cfg: DictConfig):
    # Load the CLIP model and processor

    print(OmegaConf.to_yaml(cfg))

    model = CLIPModel.from_pretrained(cfg.model.path)
    # processor = CLIPProcessor.from_pretrained(cfg.model.processor)

    # Choose computation device
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    # Load pre-trained CLIP model
    model, preprocess = clip.load(cfg.model.load, device=device, jit=False)

    if device == "cpu":
      model.float()

    #Crate dataset and dataloader
    transform = transforms.ToTensor()
    dataset = KaggleFoodDataset(csv_file=cfg.data.processed.csv, image_dir=cfg.data.processed.img, transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=1000, shuffle=True) 


    # Prepare the optimizer
    opt = cfg.model.optimizer.adam
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=opt.betas, eps=opt.eps, weight_decay=opt.wdec) # the lr is smaller, more safe for fine tuning to new dataset

    # Specify the loss function
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()


    # Initialize variable to keep track of the best loss
    best_loss = float('inf')

    # Directory where you want to save the model
    if not os.path.exists(cfg.model.save_dir):
        os.makedirs(cfg.model.save_dir)

    num_epochs = cfg.model.train.epochs

    num_epochs = cfg.model.train.epochs

    best_loss = float('inf')  # Initialize best loss to a high value
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)  # Move model to GPU at the beginning

    for epoch in range(num_epochs):
        pbar = tqdm(train_dataloader, total=len(train_dataloader), mininterval=1)  # Efficient tqdm update interval
        epoch_loss_tensor = torch.tensor(0.0, device=device)  # Accumulate loss as a GPU tensor

        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)  # Efficient zeroing of gradients

            images, texts = batch
            images = transforms.ToPILImage()(images)
            images = preprocess(images).to(device, non_blocking=True)
            texts = clip.tokenize(texts).to(device, non_blocking=True)

            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(images.size(0), dtype=torch.long, device=device)
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

            total_loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters

            epoch_loss_tensor += total_loss.detach()  # Detach the loss to prevent building up the graph

            # Update progress bar description with the current loss
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.detach():.4f}")

        # Calculate average epoch loss
        avg_epoch_loss = (epoch_loss_tensor / len(train_dataloader)).item()  # Transfer to CPU for logging

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            model_save_path = os.path.join(cfg.model.save_dir, f"{cfg.model.name}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with loss: {best_loss:.4f}")
    

if __name__ == '__main__':
    finetune_model()