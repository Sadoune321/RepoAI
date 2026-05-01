from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes.predict import router as predict_router
from app.core.config import settings

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description="API de détection de lésions cutanées — Bénin vs Malin.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api/v1")

@app.get("/")
def home():
    return {"message": "Skin Cancer API is running"}