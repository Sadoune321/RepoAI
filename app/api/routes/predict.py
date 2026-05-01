import io
import math
import base64
import torch
import torch.nn.functional as F
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont

from app.core.config import settings
from app.ml.model import get_model
from app.ml.preprocessing import preprocess_image

router = APIRouter()

FRAME_WIDTH = 14
GREEN = (34, 197, 94)
RED   = (239, 68, 68)


def _to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _add_frame(img: Image.Image, color: tuple, label: str, pct: float) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for i in range(FRAME_WIDTH):
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=color)

    band_h = 36
    draw.rectangle([0, 0, w, band_h], fill=color)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    draw.text((10, 8), f"{label}  {pct:.1f}%", fill=(255, 255, 255), font=font)
    return img


def _compute_entropy(probs: torch.Tensor) -> float:
    n = probs.shape[0]
    entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
    max_entropy = math.log(n) if n > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=415,
            detail="Format non supporté. Utilisez JPG ou PNG."
        )

    try:
        image_bytes = await file.read()

        # ── Image originale ────────────────────────────────────────────────────
        original_img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        base64_original = _to_base64(original_img)

        # ── Inférence ──────────────────────────────────────────────────────────
        img_tensor = preprocess_image(image_bytes, device=settings.DEVICE)
        model      = get_model()

        with torch.no_grad():
            logits = model(img_tensor)
            probs  = F.softmax(logits, dim=1)[0]
            pred   = torch.argmax(probs)

        prob_benign    = probs[0].item()
        prob_malignant = probs[1].item()

        is_malignant = (pred.item() == 1) or (prob_malignant >= settings.MALIGNANT_THRESHOLD)
        label        = settings.CLASSES[pred.item()]
        frame_color  = RED if is_malignant else GREEN
        confidence   = prob_malignant if is_malignant else prob_benign

        # ── Image annotée avec cadre vert/rouge ────────────────────────────────
        framed_img    = _add_frame(original_img, frame_color, label, confidence * 100)
        base64_framed = _to_base64(framed_img)

        # ── Diagnostics ────────────────────────────────────────────────────────
        entropy = _compute_entropy(probs)

        return JSONResponse({
            "status":      "success",
            "type":        "1D",
            "prediction":  label,
            "class_index": int(pred.item()),
            "confidence":  round(float(confidence), 4),
            "diagnostics": {
                "uncertainty_score": round(entropy, 4),
                "is_high_risk":      label != settings.CLASSES[0],
                "all_probabilities": {
                    settings.CLASSES[i]: round(float(probs[i]), 3)
                    for i in range(len(settings.CLASSES))
                }
            },
            "original_image":  f"data:image/png;base64,{base64_original}",
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))