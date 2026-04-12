from pydantic_settings import BaseSettings


class Settings(BaseSettings):
   
    APP_TITLE: str = "Skin Cancer Detection API"
    APP_VERSION: str = "1.0.0"

    
    DEVICE: str = "cpu"
    NUM_CLASSES: int = 7
    MODEL_PATH: str = "models/best_model.pth"

    
    CLASSES: list[str] = [
        "akiec",  # Actinic keratoses
        "bcc",    # Basal cell carcinoma
        "bkl",    # Benign keratosis
        "df",     # Dermatofibroma
        "mel",    # Melanoma
        "nv",     # Melanocytic nevus
        "vasc",   # Vascular lesion
    ]

    class Config:
        env_file = ".env"


settings = Settings()