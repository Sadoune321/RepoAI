import torch
import timm
from app.core.config import settings


_model = None


def load_model() -> torch.nn.Module:
   
    global _model
    if _model is None:
        _model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=settings.NUM_CLASSES
        )
        checkpoint = torch.load(settings.MODEL_PATH, map_location=settings.DEVICE)
        _model.load_state_dict(checkpoint)
        _model.to(settings.DEVICE)
        _model.eval()
        print(f" Modèle chargé depuis : {settings.MODEL_PATH}")
    return _model


def get_model() -> torch.nn.Module:
   
    if _model is None:
        raise RuntimeError("Le modèle n'est pas encore chargé. Appelez load_model() d'abord.")
    return _model