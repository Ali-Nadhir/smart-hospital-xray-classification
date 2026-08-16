import io
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from .inference import predict_pil_image
from .models import load_model


MODEL_NAME = os.getenv("MODEL_NAME", "mobilenet_v3")
MODEL_PATH = os.getenv("MODEL_PATH", "")
MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.5"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="X-Ray Image Classification API")
model = None


@app.on_event("startup")
def load_model_on_startup():
    global model
    if not MODEL_PATH:
        return
    path = Path(MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(f"MODEL_PATH does not exist: {path}")
    model = load_model(MODEL_NAME, path, device=DEVICE)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "device": str(DEVICE),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Set MODEL_NAME and MODEL_PATH before starting the API.",
        )

    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    return predict_pil_image(model, image, threshold=MODEL_THRESHOLD, device=DEVICE)
