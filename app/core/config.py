from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_TITLE: str = "Skin Cancer Detection API"
    APP_VERSION: str = "1.0.0"

    DEVICE: str = "cpu"
    NUM_CLASSES: int = 2
   
    MODEL_PATH: str = "models/bestModel_complete.pth"

    MALIGNANT_THRESHOLD: float = 0.50
    CLASSES: list = ["Bénin", "Malin"]

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()