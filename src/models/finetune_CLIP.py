import os

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

import clip
from src.data.CLIP_kaggle_food_dataset import KaggleFoodDataset

import hydra
from omegaconf import DictConfig

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

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
    
    def validation_step(self, batch, batch_idx):
        images, texts = batch
        images = images.to(self.device)
        texts = texts.to(self.device)

        # Forward pass
        with torch.no_grad():
            img_embs = self.model.encode_image(images)
            text_embs = self.model.encode_text(texts)
            img_embs = img_embs / torch.linalg.norm(img_embs, axis=1, keepdim=True)
            text_embs = text_embs / torch.linalg.norm(text_embs, axis=1, keepdim=True)
            logits = text_embs @ img_embs.T

        # Calculate top X percent accuracies
        top_x_percent = [0, 0.10, 0.25]
        n = logits.shape[0]
        accs = {}
    
        labels = torch.arange(n).to(self.device)
        logits_arg_sort = torch.argsort(logits, dim=1, descending=True)

        for percent_acc in top_x_percent:
            if percent_acc == 0.0:
                acc = (logits_arg_sort[:, 0] == labels).float().mean().item()
            else:
                top_x_cols = int(n * percent_acc)
                acc = (logits_arg_sort[:, :top_x_cols] == labels[:, None]).any(dim=1).float().mean().item()

            accs[f'top_{int(percent_acc * 100)}_percent_acc'] = acc

        # Logging accuracies
        for key, value in accs.items():
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=True)

        

        return accs


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.cfg.model.optimizer)
        return optimizer


@hydra.main(version_base=None, config_path=os.path.join('..', '..', 'conf'), config_name='config')
def finetune_model(cfg: DictConfig):
    #Set global seed
    L.seed_everything(42, workers=True)

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
    # Calculate the split sizes
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    test_size = total_size - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.model.train.batch_size, shuffle=True,
                              num_workers=23)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False,
                             num_workers=23)

    # Create a PyTorch Lightning module
    model = CLIPFineTuned(model, preprocess, cfg)

    # Create a PyTorch Lightning trainer and start training with DVC logging
    logger = DVCLiveLogger(save_dvc_exp=True)
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.model.save_dir, monitor='train_loss', save_top_k=1, mode='min')
    trainer = L.Trainer(max_epochs=cfg.model.train.epochs, 
                            accelerator="gpu" if torch.cuda.is_available() else "cpu",
                            deterministic=True, logger=logger,
                            callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    finetune_model()