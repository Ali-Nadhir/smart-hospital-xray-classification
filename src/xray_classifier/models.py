from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    ViT_B_16_Weights,
    mobilenet_v3_large,
    vit_b_16,
)
from torch import nn


IMAGE_SIZE = 224
CLASS_NAMES = ["normal", "abnormal"]


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def create_model(model_name: str, pretrained: bool = True):
    if model_name == "mobilenet_v3":
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)
        return model

    if model_name == "vit_b16":
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        model = vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, 1)
        return model

    raise ValueError(f"Unknown model name: {model_name}")


def load_state_dict(path: str | Path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    return checkpoint


def load_model(model_name: str, model_path: str | Path, device: str | torch.device = "cpu"):
    device = torch.device(device)
    model = create_model(model_name, pretrained=False)
    model.load_state_dict(load_state_dict(model_path))
    model.to(device)
    model.eval()
    return model


def load_image(image_path: str | Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")
