from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import timm
import io

app = FastAPI(title="Skin Cancer Detection API")

# ---------- Configuration ----------
DEVICE = "cpu"
NUM_CLASSES = 7
MODEL_PATH = "best_model.pth"   

# ---------- Load Model ----------
model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=NUM_CLASSES
)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()

# ---------- Classes ----------
classes = [
    "akiec",  # Actinic keratoses
    "bcc",    # Basal cell carcinoma
    "bkl",    # Benign keratosis
    "df",     # Dermatofibroma
    "mel",    # Melanoma
    "nv",     # Melanocytic nevus
    "vasc"    # Vascular lesion
]

# ---------- Image Preprocessing ----------
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------- Routes ----------

@app.get("/")
def home():
    return {"message": "Skin Cancer API is running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        return JSONResponse({
            "prediction": classes[pred],
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )