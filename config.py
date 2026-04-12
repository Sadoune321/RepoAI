import torch
import timm
from torchvision import transforms


DEVICE = "cpu"
NUM_CLASSES = 7
MODEL_PATH = "best_model.pth"


CLASSES = [
    "akiec",  # Actinic keratoses
    "bcc",    # Basal cell carcinoma
    "bkl",    # Benign keratosis
    "df",     # Dermatofibroma
    "mel",    # Melanoma
    "nv",     # Melanocytic nevus
    "vasc"    # Vascular lesion
]


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def load_model() -> torch.nn.Module:
    
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=NUM_CLASSES
    )
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model
