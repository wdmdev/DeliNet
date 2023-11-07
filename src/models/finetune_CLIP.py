import os

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

import clip
from transformers import CLIPModel
from src.data.kaggle_food_dataset import KaggleFoodDataset

import hydra
from omegaconf import DictConfig

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from dvclive import Live
from dvclive.lightning import DVCLiveLogger

class CLIPFineTuner(L.LightningModule):
    def __init__(self, model, preprocess, cfg):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.cfg = cfg

    def forward(self, images, texts):
        return self.model(images, texts)

    def training_step(self, batch, batch_idx):
        images, texts = batch
        texts = [str(t) for t in texts[0]]
        texts = clip.tokenize(texts).to(self.device, non_blocking=True)
        images = [transforms.ToPILImage()(image) for image in images]
        images = torch.stack([self.preprocess(image).to(self.device, non_blocking=True) for image in images])

        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()

        logits_per_image, logits_per_text = self.model(images, texts)
        ground_truth = torch.arange(images.size(0), dtype=torch.long, device=self.device)
        img_loss = loss_img(logits_per_image, ground_truth)
        txt_loss = loss_txt(logits_per_text, ground_truth)
        total_loss = (img_loss + txt_loss) / 2

        self.log("train_img_loss", img_loss, on_step=False, on_epoch=True, batch_size=self.cfg.model.train.batch_size)
        self.log("train_txt_loss", txt_loss, on_step=False, on_epoch=True, batch_size=self.cfg.model.train.batch_size)
        self.log("train_loss", total_loss, on_step=False, on_epoch=True, batch_size=self.cfg.model.train.batch_size)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.cfg.model.optimizer)
        return optimizer


@hydra.main(version_base=None, config_path=os.path.join('..', '..', 'conf'), config_name='config')
def finetune_model(cfg: DictConfig):
    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained(cfg.model.path)
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    model, preprocess = clip.load(cfg.model.load, device=device, jit=False)

    if device == "cpu":
      model.float()

    # Create dataset and dataloader
    transform = transforms.ToTensor()
    dataset = KaggleFoodDataset(csv_file=cfg.data.processed.csv, 
                                image_dir=cfg.data.processed.img, transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=cfg.model.train.batch_size, 
                                    shuffle=True, num_workers=23) 

    # Create a PyTorch Lightning module
    L.seed_everything(42, workers=True)
    model = CLIPFineTuner(model, preprocess, cfg)

    # Create a PyTorch Lightning trainer and start training with DVC logging
    logger = DVCLiveLogger(save_dvc_exp=True)
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.model.save_dir, monitor='train_loss', save_top_k=1, mode='min')
    trainer = L.Trainer(max_epochs=cfg.model.train.epochs, 
                            accelerator="gpu" if torch.cuda.is_available() else "cpu",
                            deterministic=True, logger=logger,
                            callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloader)
    trainer.save_

if __name__ == '__main__':
    finetune_model()