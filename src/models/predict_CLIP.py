import os
import torch
import requests
from PIL import Image
from src.models.finetune_CLIP import CLIPFineTuned
from torchvision import transforms

import clip

import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path=os.path.join('..', '..', 'conf'), config_name='config')
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu" 


    # Load the CLIP model and processor
    model, preprocess = clip.load(cfg.model.load, device=device, jit=False)

    #Load the state_dict from the checkpoint
    checkpoint = torch.load("models/CLIP1/epoch=29-step=420.ckpt")
    model = CLIPFineTuned(model, preprocess, cfg)
    model.load_state_dict(checkpoint['state_dict'])

    # url = "https://th.bing.com/th/id/OIP.b7gaP88IQPpxPbQBVgLN-wHaF-?pid=ImgDet&rs=1" #hotdog
    url = "https://thedumplingsisters.files.wordpress.com/2014/11/tds-noodles-2.jpg" #wok noodles
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    image = transforms.Resize((224,224))(image)
    image = preprocess(image).unsqueeze(0).to(device)
    
    labels =['soup', 'ice cream', 'pizza', 'hotdog', 'noodles'] 
    text = clip.tokenize(labels).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    for label, prob in zip(labels, probs[0]):
        print(f"{label}: {prob}")

if __name__ == '__main__':
    main()