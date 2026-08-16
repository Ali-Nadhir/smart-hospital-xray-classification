from pathlib import Path

import torch

from .models import CLASS_NAMES, build_transform, load_image, load_model


@torch.no_grad()
def predict_pil_image(model, image, threshold: float = 0.5, device: str | torch.device = "cpu"):
    transform = build_transform()
    device = torch.device(device)

    tensor = transform(image).unsqueeze(0).to(device)
    probability_abnormal = torch.sigmoid(model(tensor)).item()

    label_index = int(probability_abnormal >= threshold)
    prediction = CLASS_NAMES[label_index]
    confidence = probability_abnormal if label_index == 1 else 1.0 - probability_abnormal

    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "threshold": float(threshold),
        "probability_normal": float(1.0 - probability_abnormal),
        "probability_abnormal": float(probability_abnormal),
    }


def predict_image(
    model_name: str,
    model_path: str | Path,
    image_path: str | Path,
    threshold: float = 0.5,
    device: str | torch.device = "cpu",
):
    model = load_model(model_name, model_path, device=device)
    image = load_image(image_path)
    return predict_pil_image(model, image, threshold=threshold, device=device)
