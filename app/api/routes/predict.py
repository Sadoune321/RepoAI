import io
import base64
import torch
import torch.nn.functional as F
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw

from app.core.config import settings
from app.ml.model import get_model
from app.ml.preprocessing import preprocess_image

router = APIRouter()

FRAME_WIDTH = 14
GREEN = (34, 197, 94)   
RED   = (239, 68, 68)   


def _add_frame(image_bytes: bytes, color: tuple, label: str, pct: float) -> str:
    
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Cadre
    for i in range(FRAME_WIDTH):
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=color)

    # Bandeau texte
    band_h = 30
    draw.rectangle([0, 0, w, band_h], fill=color)
    draw.text((10, 7), f"{label}  {pct:.1f}%", fill=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=415,
            detail="Format non supporté. Utilisez JPG ou PNG."
        )

    try:
        image_bytes = await file.read()

       
        img_tensor = preprocess_image(image_bytes, device=settings.DEVICE)
        model = get_model()

        with torch.no_grad():
            logits = model(img_tensor)                      
            probs  = F.softmax(logits, dim=1)[0]            
            pred   = torch.argmax(probs).item()              

        prob_benign    = probs[0].item()
        prob_malignant = probs[1].item()

        
        is_malignant = (pred == 1) or (prob_malignant >= settings.MALIGNANT_THRESHOLD)
        label        = "Malin" if is_malignant else "Bénin"
        frame_color  = RED if is_malignant else GREEN
        confidence   = prob_malignant if is_malignant else prob_benign

        framed_b64 = _add_frame(image_bytes, frame_color, label, confidence * 100)

        return JSONResponse({
            "prediction":       label,
            "prob_benign":      round(prob_benign, 4),
            "prob_malignant":   round(prob_malignant, 4),
            "confidence":       round(confidence, 4),
            "threshold_used":   settings.MALIGNANT_THRESHOLD,
            "frame_color":      "red" if is_malignant else "green",
            "framed_image_b64": framed_b64
            
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))