import io
import torch
from PIL import Image
from torchvision import transforms


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  
        std=[0.229, 0.224, 0.225]
    ),
])


def preprocess_image(image_bytes: bytes, device: str = "cpu") -> torch.Tensor:

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(device)
    return tensor