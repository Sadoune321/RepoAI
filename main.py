import io
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from config import CLASSES, DEVICE, TRANSFORM, load_model

app = FastAPI(title="Skin Cancer Detection API")


model = load_model()


@app.get("/")
def home():
    return {"message": "Skin Cancer API is running "}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        
        img_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

       
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        return JSONResponse({
            "prediction": CLASSES[pred],
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )
