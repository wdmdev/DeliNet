import json
import torch
import clip

from src.data.food101_dataset import Food101Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

#CLIP model
from src.models.model_utils import CLIP_vision_wrapper, CLIP_text_wrapper

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Download and load the Food101 dataset
data_transforms = Compose([Resize(224), CenterCrop(224), ToTensor(), 
                           Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
food101_dataset = Food101Dataset(data_dir='data/raw/food-101/food-101/images', transform=data_transforms)

# Combine train and test datasets
data_loader = DataLoader(food101_dataset, batch_size=32, shuffle=False, num_workers=23)

# Prepare text tokens
class_names = food101_dataset.classes  # Assuming train and test have the same classes
augmented_class_names = [f"a photo of {c}, a type of food." for c in class_names]

# Function to compute accuracy
def compute_accuracy(data_loader, vision_model, text_model, classes):
    vision_model.eval()
    text_model.eval()
    total = 0
    correct = 0

    # class_features = text_model(class_names)
    class_features = text_model(classes)
    

    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Calculate features
            image_features = vision_model(images)

            # Calculate similarities between image features and text features
            similarities = torch.matmul(image_features, class_features.T)

            # Get the indices of the classes that have the highest similarity
            predictions = similarities.argmax(dim=-1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    return correct / total

# Compute accuracies
results = {}
vision_model = torch.load("models/EfficientTrans_AND_DistilBert_3xNet_TrivialAugmentWide/EfficientTrans.pt").to(device)
text_model = torch.load("models/EfficientTrans_AND_DistilBert_3xNet_TrivialAugmentWide/DistilBert_mono.pt").to(device)
print("##################### Our Model ####################")
classname_accuracy = compute_accuracy(data_loader, vision_model, text_model, class_names)
print(f"Standard Class Names, Our Model Accuracy: {classname_accuracy:.2f}")
aug_class_accuracy = compute_accuracy(data_loader, vision_model, text_model, augmented_class_names)
print(f"Augmented Class Names, Our Model Accuracy: {aug_class_accuracy:.2f}")
results["Our Model"] = {
    "Standard Class Names Accuracy": classname_accuracy,
    "Augmented Class Names Accuracy": aug_class_accuracy
}


vision_model = CLIP_vision_wrapper(d=device).to(device)
text_model = CLIP_text_wrapper(d=device).to(device)
print("########## CLIP ##############")
classname_accuracy = compute_accuracy(data_loader, vision_model, text_model, class_names)
print(f"Standard Class Names: {classname_accuracy:.2f}")
aug_class_accuracy = compute_accuracy(data_loader, vision_model, text_model, augmented_class_names)
print(f"Augmented Class Names: {aug_class_accuracy:.2f}")
results["CLIP"] = {
    "Standard Class Names Accuracy": classname_accuracy,
    "Augmented Class Names Accuracy": aug_class_accuracy
}

with open('results.json', 'w') as f:
    json.dump(results, f)

