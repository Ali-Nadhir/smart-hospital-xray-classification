# AI-Based X-Ray Image Classification System

AI healthcare module for binary musculoskeletal X-ray classification on the **Stanford MURA v1.1 dataset**, developed for integration into a **Smart Hospital Management System**.

---

## 📌 Project Overview

The system classifies musculoskeletal radiographs into:
* `normal` (no fracture / lesion / structural abnormality)
* `abnormal` (fracture, hardware, degenerative joint disease, lesion)

### Evaluated Architectures:
1. **MobileNetV3 Large** (Lightweight Convolutional Neural Network baseline)
2. **Vision Transformer (ViT-B/16)** (Self-Attention Transformer baseline)

---

## 📊 Final Benchmark Results

Evaluated on the complete MURA v1.1 validation split (3,197 radiograph images across 7 anatomical regions):

| Model | Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Optimal Threshold ($\tau$) | Parameters | Size (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3 Large** 🏆 | **CNN** | **82.36%** | **82.03%** | **80.85%** | **81.44%** | **0.8824** | **`0.33`** | **4.2M** | **16.8 MB** |
| **ViT-B/16** | **Transformer** | 80.26% | 78.58% | 80.78% | 79.66% | 0.8721 | `0.47` | 85.8M | 330.3 MB |

> **Key Takeaway:** MobileNetV3 Large outperforms ViT-B/16 by **+1.78% F1** and **+0.0103 ROC-AUC** while being **$20\times$ smaller** and **$10\times$ faster during inference**. It is the recommended model for real-time clinical triage.
>
> Full analysis available in [`docs/evaluation_report.md`](docs/evaluation_report.md).

---

## 📁 Repository Structure

```text
.
├── 01_mura_xray_mobilenet_vit_colab.ipynb   # Complete training & evaluation notebook
├── src/
│   └── xray_classifier/
│       ├── __init__.py
│       ├── api.py                          # FastAPI prediction microservice
│       ├── dataset.py                      # MURA dataset loaders & transforms
│       ├── inference.py                    # Inference & confidence scoring
│       └── models.py                       # MobileNetV3 & ViT-B/16 architectures
├── scripts/
│   └── predict_image.py                    # CLI prediction tool
├── docs/
│   └── evaluation_report.md                # Comprehensive medical evaluation report
├── requirements.txt                        # Project dependencies
├── pyproject.toml                          # Package configuration
└── README.md
```

---

## 🚀 Getting Started Locally

### 1. Installation

```bash
# Clone the repository
git clone <YOUR_REPO_URL>
cd smart-hospital-xray-classification

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # On Windows
# source .venv/bin/activate  # On Linux/macOS

# Install dependencies and local package
pip install -r requirements.txt
pip install -e .
```

### 2. Download Model Weights

Download the trained model checkpoints and place them in the `models/` directory:

* [MobileNetV3 Large Best Weights (`mobilenet_v3_best.pt`)](https://drive.google.com/file/d/11zax6-BWNNbk5ylz2NwibNLG2EFs_nBO/view?usp=sharing) *(16.8 MB)*
* [ViT-B/16 Best Weights (`vit_b16_best.pt`)](https://drive.google.com/file/d/13dmEhKIGmc2yuyqZPlAK0yFaTm3DQWoJ/view?usp=sharing) *(330.3 MB)*

```text
models/
├── mobilenet_v3_best.pt
└── vit_b16_best.pt
```

---

## 🔍 Single Image CLI Prediction

Run predictions on any single X-ray image with calibrated thresholds:

### Using MobileNetV3 Large (Recommended):
```bash
python scripts/predict_image.py \
  --model-name mobilenet_v3 \
  --model-path models/mobilenet_v3_best.pt \
  --image path/to/xray.png \
  --threshold 0.33
```

### Using ViT-B/16:
```bash
python scripts/predict_image.py \
  --model-name vit_b16 \
  --model-path models/vit_b16_best.pt \
  --image path/to/xray.png \
  --threshold 0.47
```

#### Example Output:
```json
{
  "prediction": "abnormal",
  "confidence": 0.8842,
  "threshold": 0.33,
  "probability_normal": 0.1158,
  "probability_abnormal": 0.8842
}
```

---

## 🌐 FastAPI Smart Hospital API Integration

Start the prediction API service:

```bash
# Set environment variables (Windows CMD)
set MODEL_NAME=mobilenet_v3
set MODEL_PATH=models/mobilenet_v3_best.pt
set MODEL_THRESHOLD=0.33

# Or on Linux / macOS
# export MODEL_NAME=mobilenet_v3
# export MODEL_PATH=models/mobilenet_v3_best.pt
# export MODEL_THRESHOLD=0.33

# Launch server
uvicorn xray_classifier.api:app --reload --port 8000
```

### Endpoints:
* `GET /health` - Service health & model status.
* `POST /predict` - Upload X-ray image file (`multipart/form-data`, key: `file`).

Interactive API Swagger documentation is available at `http://localhost:8000/docs`.

---

## 🔬 Training Pipeline on Google Colab

The training notebook [`01_mura_xray_mobilenet_vit_colab.ipynb`](01_mura_xray_mobilenet_vit_colab.ipynb) includes:
* Automated Kaggle dataset download and fast local SSD extraction.
* Google Drive checkpoint caching and automatic session resume (`last_checkpoint.pt`).
* Class-weighted loss (`BCEWithLogitsLoss`) for imbalanced clinical data.
* Dynamic validation F1 threshold grid search ($0.05 \dots 0.95$).
* Confusion matrix visualization and performance export (`mobilenet_vs_vit_metrics.csv`).

---

## ⚠️ Clinical Disclaimer

This software is an AI decision-support prototype intended for research, screening prioritization, and workflow efficiency. It does not replace professional radiological diagnosis. All clinical findings must be verified by a licensed medical practitioner.
