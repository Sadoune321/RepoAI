import torch
import torch.nn as nn
import timm
import sys
from app.core.config import settings

_model = None


class SkinClassifier(nn.Module):
   
    def __init__(self, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b3", pretrained=False, num_classes=0
        )
        in_features = self.backbone.num_features  # 1536

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)



sys.modules['__main__'].SkinClassifier = SkinClassifier


def get_model() -> nn.Module:
    
    global _model
    if _model is None:
        print(f"Chargement du modèle: {settings.MODEL_PATH}")
        
       
        try:
            _model = torch.load(
                settings.MODEL_PATH,
                map_location=settings.DEVICE
            )
            print(" Modèle chargé (format complet avec classe)")
            
        except AttributeError as e:
            print(f" Erreur de classe: {e}")
            print("Tentative avec unpickler personnalisé...")
            
            
            import pickle
            
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if name == 'SkinClassifier':
                        return SkinClassifier
                    return super().find_class(module, name)
            
            with open(settings.MODEL_PATH, 'rb') as f:
                _model = CustomUnpickler(f).load()
        
        _model.to(settings.DEVICE)
        _model.eval()
        print(f" Modèle prêt sur {settings.DEVICE}")
        
    return _model