import torch
from torch.nn.functional import triplet_margin_loss
from torch.nn.functional import cross_entropy

def triplet_loss(text_latent, image_latent, labels, text_model=None):
  n,m = text_latent.shape
  while True:
    perm = torch.randperm(n, device=text_latent.get_device())
    if not torch.any(labels == perm):
      break

  loss = triplet_margin_loss(image_latent, text_latent, text_latent[perm])

  return loss

def contrastive_loss(text_latent, img_latent, labels, text_model):
  logits = (text_latent @ img_latent.T) * torch.exp(text_model.t)

  loss_i = cross_entropy(logits, labels)
  loss_t = cross_entropy(logits.T, labels)
  loss = (loss_i + loss_t) / 2.0

  return loss
