from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from io import BytesIO
import base64
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import os

app = FastAPI(title="CNN Image Classifier")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def read_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

# Load model
model = load_model("CNN_model_tuned2.keras")
class_names = ["drink", "food", "inside", "menu", "outside"]  # Adjust as needed

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image = image.resize((128, 128))
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        pred_idx = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        
         # ---- LIME Explanation ----
        explainer = lime_image.LimeImageExplainer()

        def predict_fn(images):
            images = np.array(images)
            return model.predict(images)

        explanation = explainer.explain_instance(
            img_array[0],                      # single image (no batch)
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=1000
        )

        lime_img, mask = explanation.get_image_and_mask(
            label=explanation.top_labels[0],
            positive_only=True,
            hide_rest=False,
            num_features=5
        )

        # Save and encode LIME image
        plt.imshow(mark_boundaries(lime_img, mask))
        plt.axis('off')
        lime_path = "static/lime_explanation.png"
        plt.savefig(lime_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        with open(lime_path, "rb") as f:
            lime_base64 = base64.b64encode(f.read()).decode()

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return JSONResponse(content={
            "predicted_class_index": int(pred_idx),
            "predicted_class_label": class_names[pred_idx],
            "confidence": round(confidence, 3),
            "image_base64": img_base64,
            "lime_base64": lime_base64
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
