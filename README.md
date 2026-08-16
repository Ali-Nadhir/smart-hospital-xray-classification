# X-Ray Image Classification System

AI module for binary musculoskeletal X-ray classification using the MURA v1.1 dataset.

The project compares:

- MobileNetV3
- Vision Transformer ViT-B/16

Classes:

- `normal`
- `abnormal`

Training is designed for Google Colab. Local code is mainly for inference, API serving, and repository submission.

## Project Structure

```text
.
├── 01_mura_xray_mobilenet_vit_colab.ipynb
├── src/
│   └── xray_classifier/
│       ├── __init__.py
│       ├── api.py
│       ├── dataset.py
│       ├── inference.py
│       └── models.py
├── scripts/
│   └── predict_image.py
├── requirements.txt
└── README.md
```

## Training

Use the Colab notebook:

```text
01_mura_xray_mobilenet_vit_colab.ipynb
```

Current training defaults:

```python
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3
MODEL_LEARNING_RATES = {
    "mobilenet_v3": 1e-4,
    "vit_b16": 1e-5,
}
```

The notebook saves checkpoints and history to Google Drive:

```text
/content/drive/MyDrive/mura_xray_outputs
```

Important files:

```text
mobilenet_v3_best.pt
mobilenet_v3_last_checkpoint.pt
mobilenet_v3_history.csv
vit_b16_best.pt
vit_b16_last_checkpoint.pt
vit_b16_history.csv
mobilenet_vs_vit_metrics.csv
```

Use `best.pt` files for inference. Use `last_checkpoint.pt` files only to resume training.

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Place downloaded model weights in a local folder such as:

```text
models/mobilenet_v3_best.pt
models/vit_b16_best.pt
```

## Single Image Prediction

```bash
python scripts/predict_image.py --model-name mobilenet_v3 --model-path models/mobilenet_v3_best.pt --image path/to/xray.png --threshold 0.34
```

For ViT:

```bash
python scripts/predict_image.py --model-name vit_b16 --model-path models/vit_b16_best.pt --image path/to/xray.png --threshold 0.54
```

Use the threshold from the notebook evaluation table.

## API

Configure model loading with environment variables:

```bash
set MODEL_NAME=mobilenet_v3
set MODEL_PATH=models/mobilenet_v3_best.pt
set MODEL_THRESHOLD=0.34
```

Run:

```bash
uvicorn xray_classifier.api:app --reload
```

Endpoint:

```text
POST /predict
```

Upload an image file using form-data key:

```text
file
```

## Notes

This project is a decision-support prototype, not a clinical diagnostic system. All predictions must be reviewed by qualified medical professionals.
