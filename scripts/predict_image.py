import argparse
import json

import torch

from xray_classifier.inference import predict_image


def parse_args():
    parser = argparse.ArgumentParser(description="Predict normal/abnormal for one X-ray image.")
    parser.add_argument("--model-name", choices=["mobilenet_v3", "vit_b16"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    result = predict_image(
        model_name=args.model_name,
        model_path=args.model_path,
        image_path=args.image,
        threshold=args.threshold,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
