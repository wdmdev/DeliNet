import os

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

import clip
from src.data.CLIP_kaggle_food_dataset import KaggleFoodDataset

import hydra
from omegaconf import DictConfig

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import CLIPModel

from dvclive.lightning import DVCLiveLogger

class CLIPFineTuned(L.LightningModule):
    def __init__(self, model=None, preprocess=None, cfg=None):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.cfg = cfg
        self.automatic_optimization = False #set for gradient accumulation

    def forward(self, images, texts):
        return self.model(images, texts)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        images, texts = batch
        images = images.to(self.device)
        texts = texts.to(self.device)

        max_batch_size = self.cfg.model.train.max_batch_part_size
        num_images = images.size(0)
        num_splits = (num_images + max_batch_size - 1) // max_batch_size

        total_loss_i = 0
        total_loss_t = 0

        for i in range(num_splits):
            start = i * max_batch_size
            end = min(start + max_batch_size, num_images)

            batch_images = images[start:end]
            batch_texts = texts[start:end]

            #cosine similarity as logits
            logits_image, logits_text = self.model(batch_images, batch_texts)

            # symmetric loss function
            ground_truth = torch.arange(batch_images.size(0), dtype=torch.long, device=self.device)
            loss_i = F.cross_entropy(logits_image, ground_truth)
            loss_t = F.cross_entropy(logits_text, ground_truth)
            total_loss_i += loss_i
            total_loss_t += loss_t

            # Backward pass
            loss = (loss_i + loss_t) / 2
            self.manual_backward(loss)
        
            # Update parameters every num_splits batches
            if (i + 1) % num_splits == 0:
                opt.step()
                opt.zero_grad()

        avg_loss_i = total_loss_i / num_splits
        avg_loss_t = total_loss_t / num_splits
        avg_loss = (avg_loss_i + avg_loss_t) / 2

        self.log("train_img_loss", avg_loss_i, on_step=False, on_epoch=True, batch_size=self.cfg.model.train.batch_size)
        self.log("train_txt_loss", avg_loss_t, on_step=False, on_epoch=True, batch_size=self.cfg.model.train.batch_size)
        self.log("train_loss", avg_loss, on_step=False, on_epoch=True, batch_size=self.cfg.model.train.batch_size)

        return avg_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.cfg.model.optimizer)
        return optimizer


@hydra.main(version_base=None, config_path=os.path.join('..', '..', 'conf'), config_name='config')
def finetune_model(cfg: DictConfig):
    # Load the CLIP model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    model, preprocess = clip.load(cfg.model.load, device=device, jit=False)

    if device == "cpu":
      model.float()

    def transform(img):
        img = img.convert("RGB")
        img = transforms.Resize((224,224))(img)
        img = preprocess(img)

        return img

    dataset = KaggleFoodDataset(csv_file=cfg.data.processed.csv, 
                                image_dir=cfg.data.processed.img, transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=cfg.model.train.batch_size, 
                                    shuffle=True, num_workers=23) 

    # Create a PyTorch Lightning module
    L.seed_everything(42, workers=True)
    model = CLIPFineTuned(model, preprocess, cfg)

    # Create a PyTorch Lightning trainer and start training with DVC logging
    logger = DVCLiveLogger(save_dvc_exp=True)
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.model.save_dir, monitor='train_loss', save_top_k=1, mode='min')
    trainer = L.Trainer(max_epochs=cfg.model.train.epochs, 
                            accelerator="gpu" if torch.cuda.is_available() else "cpu",
                            deterministic=True, logger=logger,
                            callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloader)

if __name__ == '__main__':
    finetune_model()