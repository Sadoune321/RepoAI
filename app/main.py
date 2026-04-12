from fastapi import FastAPI
from app.api.routes.predict import router as predict_router
from app.core.config import settings
from app.ml.model import load_model

app = FastAPI(title=settings.APP_TITLE)

model = load_model()

app.include_router(predict_router, prefix="/api/v1")

@app.get("/")
def home():
    return {"message": "Skin Cancer API is running "}