from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from torchvision import models, transforms
from PIL import Image
import torch
from io import BytesIO
import base64

app = FastAPI(title="Fine-tuned VGG16 Classifier")

# Mount static files (for index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")



# Serve index.html at root
@app.get("/", response_class=HTMLResponse)
def read_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load fine-tuned VGG16
model = models.vgg16(weights=None)
model.classifier[6] = torch.nn.Linear(4096, 5)
model.load_state_dict(torch.load("vgg_finetuned_round2.pth", map_location=device))
model.to(device)
model.eval()

# Label mapping
idx_to_label = {
    0: "drink",
    1: "food",
    2: "inside",
    3: "menu",
    4: "outside"
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Load and preprocess image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            pred_idx = torch.argmax(probabilities, 1).item()
            pred_label = idx_to_label.get(pred_idx, "Unknown")
            confidence = round(probabilities[0][pred_idx].item(), 3)


        # Encode image to base64 to return in response
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return JSONResponse(content={
    "predicted_class_index": pred_idx,
    "predicted_class_label": pred_label,
    "confidence": confidence,
    "image_base64": img_base64
})


    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
