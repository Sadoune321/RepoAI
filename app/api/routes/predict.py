import torch
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.ml.model import get_model
from app.ml.preprocessing import preprocess_image

router = APIRouter()


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
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        return JSONResponse({
            "prediction": settings.CLASSES[pred],
            "confidence": round(confidence, 4),
            "all_scores": {
                cls: round(probs[0][i].item(), 4)
                for i, cls in enumerate(settings.CLASSES)
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))