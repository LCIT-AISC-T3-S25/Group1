import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "./model/cgan_complete.pth")  
    LATENT_DIM = int(os.getenv("LATENT_DIM", 100))
    IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 64))
    LABEL_DIM = 4          # number of classes for conditioning
    IMG_CHANNELS = 3       # RGB images
    FEATURE_DIM = 64
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8002))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

config = Config()
