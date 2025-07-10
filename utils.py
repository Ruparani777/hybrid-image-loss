import torch
import torchvision.transforms as transforms
from PIL import Image

def load_image(path, size=(256, 256)):
    image = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)
