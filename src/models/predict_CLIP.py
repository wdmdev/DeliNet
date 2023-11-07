import os
import torch
import requests
from PIL import Image
from src.models.finetune_CLIP import CLIPFineTuned
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms

import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path=os.path.join('..', '..', 'conf'), config_name='config')
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    #Load the state_dict from the checkpoint
    checkpoint = torch.load("models/CLIP1/epoch=0-step=54.ckpt")
    state_dict = checkpoint['state_dict']

    # Load the model from Hugging Face and apply the state_dict
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", state_dict=state_dict)
    model = model.to(device)
    model.eval()

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    url = "https://images7.alphacoders.com/596/596343.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    image = transforms.Resize((224,224))(image)

    labels =['pasta', 'soup', 'sandwich', 'pizza'] 
    inputs = processor(text=labels, images=image, return_tensors='pt')
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    probs_list = probs.tolist()[0]
    for label, prob in zip(labels, probs_list):
        print(f"{label}: {prob}")

if __name__ == '__main__':
    main()