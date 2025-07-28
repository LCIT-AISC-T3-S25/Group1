from fastapi import FastAPI
from fastapi.responses import FileResponse
from .model_architecture import Generator
from .utils import generate_image
from .config import config
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(
    noise_dim=config.LATENT_DIM,
    label_dim=4,
    img_channels=3,
    feature_dim=64
)

# ✅ Load checkpoint safely
checkpoint = torch.load(config.MODEL_PATH, map_location=device)

# Load only generator weights if checkpoint contains multiple keys
if "generator_state_dict" in checkpoint:
    generator.load_state_dict(checkpoint["generator_state_dict"])
else:
    generator.load_state_dict(checkpoint)

generator.to(device).eval()

app = FastAPI()

@app.get("/")
def home():
    return {"message": "CGAN API running"}

@app.get("/generate")
def generate(label: int = 0):
    img_path = generate_image(generator, config.LATENT_DIM, label, device, output_path="generated.png")
    return FileResponse(img_path, media_type="image/png")
