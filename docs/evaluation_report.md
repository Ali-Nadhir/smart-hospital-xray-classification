# X-Ray Image Classification Model Evaluation Report

**Project:** AI-Based X-Ray Image Classification System Using Deep Learning  
**Dataset:** Stanford MURA v1.1 (Musculoskeletal Radiographs)  
**Task:** Binary Classification (`0 = normal`, `1 = abnormal`)  
**Target Application:** Smart Hospital Management System AI Screening Module  

---

## 1. Executive Summary

This report provides the comparative evaluation of two deep learning architectures for musculoskeletal radiograph abnormality classification on the MURA v1.1 dataset:
1. **MobileNetV3 Large** (Lightweight Convolutional Neural Network baseline)
2. **Vision Transformer ViT-B/16** (Self-attention Transformer baseline)

Both models were trained and evaluated on the full MURA v1.1 dataset ($\approx 36,808$ training images across elbow, finger, forearm, hand, humerus, shoulder, and wrist studies). 

**Key Findings:**
* **MobileNetV3 Large achieved the highest overall performance**, reaching an **F1-Score of 81.44%**, **Accuracy of 82.36%**, and an **ROC-AUC of 0.8824** at an optimal decision threshold of **`0.33`**.
* **ViT-B/16 achieved solid performance** (**F1: 79.66%**, **Accuracy: 80.26%**, **ROC-AUC: 0.8721** at threshold `0.47`), but required $\approx 5\times$ more training time per epoch and produced a $\approx 20\times$ larger model binary ($330.3\text{ MB}$ vs. $16.8\text{ MB}$).
* **Primary Recommendation:** MobileNetV3 Large is the recommended deployment model for real-time triage in the Smart Hospital Management System due to its superior clinical accuracy, minimal latency, and lightweight resource footprint.

---

## 2. Training Configuration & Methodology

| Parameter / Setting | Value | Technical Justification |
| :--- | :--- | :--- |
| **Dataset** | MURA v1.1 | 40,000+ images across 7 musculoskeletal body parts |
| **Input Resolution** | $224 \times 224$ (RGB) | Standard resolution for ImageNet-pretrained transfer learning |
| **Batch Size** | `32` | Matches documentation; balances gradient stability and GPU memory |
| **Optimizer** | AdamW | Weight decay regularizes parameter updates |
| **Loss Function** | `BCEWithLogitsLoss` | Class-weighted loss ($w_{\text{pos}} = 1.4748$) penalizes missed fractures |
| **MobileNetV3 LR** | `1e-4` ($0.0001$) | Standard fine-tuning learning rate for CNN transfer learning |
| **ViT-B/16 LR** | `1e-5` ($0.00001$) | Stable fine-tuning rate preventing self-attention disruption |
| **Data Augmentation** | Random rotation ($\pm 10^\circ$), horizontal flip, resize, ImageNet normalization | Prevents memorization of scanner artifacts |
| **Decision Cutoff** | Dynamic Validation F1 Grid Search | Optimizes threshold $\tau \in [0.05, 0.95]$ per architecture |

---

## 3. Comparative Evaluation Results

The table below summarizes the peak validation performance across both models on the full MURA validation split (3,197 images):

| Metric | MobileNetV3 Large (CNN) | ViT-B/16 (Transformer) | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Validation F1-Score** | **`0.8144` (81.44%)** | `0.7966` (79.66%) | **+1.78% (MobileNet)** |
| **Validation Accuracy** | **`0.8236` (82.36%)** | `0.8026` (80.26%) | **+2.10% (MobileNet)** |
| **Validation Precision** | **`0.8203` (82.03%)** | `0.7858` (78.58%) | **+3.45% (MobileNet)** |
| **Validation Recall (Sensitivity)** | **`0.8085` (80.85%)** | `0.8078` (80.78%) | **+0.07% (MobileNet)** |
| **Validation ROC-AUC** | **`0.8824`** | `0.8721` | **+0.0103 (MobileNet)** |
| **Validation Loss** | `0.5672` | `0.5359` | -0.0313 (ViT) |
| **Optimal Threshold ($\tau$)** | **`0.33`** | `0.47` | — |
| **Trainable Parameters** | **$4,203,313$ ($\approx 4.2\text{M}$)** | $85,799,425$ ($\approx 85.8\text{M}$)** | **$20.4\times$ smaller** |
| **Model Size on Disk** | **`16.8 MB`** | `330.3 MB` | **$19.7\times$ smaller** |
| **Training Speed** | **$\approx 5\text{m }50\text{s}$ / epoch** | $\approx 28\text{m }10\text{s}$ / epoch | **$4.8\times$ faster** |
| **Inference Latency (GPU)** | **$\approx 2\text{ ms}$ / image** | $\approx 20\text{ ms}$ / image | **$10\times$ faster** |

---

## 4. In-Depth Analysis & Architectural Comparison

### A. Predictive Performance & Inductive Bias
* **Why MobileNetV3 outperformed ViT-B/16:**
  Convolutional networks have inherent **inductive biases** (locality and translation equivariance) that are naturally suited for 2D radiological images where structural landmarks (cortical bone edges, joint spaces) have strict spatial locality.
  Vision Transformers lack these local inductive biases and rely purely on global self-attention across image patches ($14 \times 14 = 196$ patches). On a medium-sized dataset ($\approx 37\text{k}$ images), the CNN learns robust discriminative features much faster without requiring hundreds of thousands of pretraining samples.

### B. Sensitivity & Clinical Safety (Recall)
* In clinical radiology screening, **Recall (Sensitivity)** is paramount: missing a fracture (False Negative) has severe clinical consequences compared to a false alarm (False Positive).
* MobileNetV3 achieved **$80.85\%$ Recall** with a calibrated threshold of **`0.33`**. This threshold ensures that the model aggressively flags subtle abnormalities while maintaining an **$82.03\%$ Precision**.

### C. Resource & Latency Footprint
* MobileNetV3 requires only **$16.8\text{ MB}$** of storage and **$4.2\text{M}$ parameters**, making it ideal for edge deployment, microservices, or integration into existing hospital picture archiving and communication systems (PACS).
* ViT-B/16 requires **$330.3\text{ MB}$** and heavy matrix multiplications, resulting in $10\times$ higher inference latency without providing any accuracy benefit on this task.

### D. Training Dynamics, Capacity & Empirical Overfitting Evidence
* **MobileNetV3 (CNN):** Converged rapidly and reached peak generalization at **Epoch 4** (Validation F1: `0.8144`, ROC-AUC: `0.8824`). Subsequent epochs (5–10) showed mild over-tuning where training loss dropped from `0.463` to `0.283` (90.2% train accuracy) while validation loss crept from `0.567` to `0.775`. The checkpointing system automatically isolated and preserved the optimal Epoch 4 weights.
* **ViT-B/16 (Transformer):** Reached its optimal validation performance early at **Epoch 2** (Validation F1: `0.7966`, ROC-AUC: `0.8721`, Loss: `0.5359`). Additional training across Epochs 6 and 7 provided conclusive empirical evidence of the transformer capacity plateau:
  * By **Epoch 7**, training loss dropped to `0.2746` ($91.15\%$ training accuracy), but validation loss climbed sharply by $+35\%$ to `0.7220` and validation ROC-AUC degraded to `0.8502`.
  * Because the checkpointing callback tracks peak validation F1, `vit_b16_best.pt` remained locked to the optimal Epoch 2 weights without degradation.

### E. Confusion Matrix Analysis (Validation Set: 3,197 Studies)
* **MobileNetV3 (Threshold $\tau = 0.33$):**
  * **True Positives (Abnormal caught):** $1,237 / 1,530$ ($\approx 80.85\%$ Sensitivity)
  * **True Negatives (Normal cleared):** $1,396 / 1,667$ ($\approx 83.74\%$ Specificity)
  * **False Positives (False alarms):** $271$
  * **False Negatives (Missed abnormalities):** $293$
* **Clinical Assessment:** The calibrated threshold strikes the necessary clinical safety balance by minimizing missed structural lesions while maintaining high operational throughput for screening radiologists.

---

## 5. Model Weights & Checkpoints

The trained weights are stored and downloadable from Google Drive:

* **MobileNetV3 Best Weights (`mobilenet_v3_best.pt`):**  
  [Download from Google Drive](https://drive.google.com/file/d/11zax6-BWNNbk5ylz2NwibNLG2EFs_nBO/view?usp=sharing) *(File size: ~16.8 MB)*
* **ViT-B/16 Best Weights (`vit_b16_best.pt`):**  
  [Download from Google Drive](https://drive.google.com/file/d/13dmEhKIGmc2yuyqZPlAK0yFaTm3DQWoJ/view?usp=sharing) *(File size: ~330.3 MB)*

To run inference locally, place the `.pt` files into the `models/` directory:
```text
models/
├── mobilenet_v3_best.pt
└── vit_b16_best.pt
```

---

## 6. Clinical Limitations & Ethical Considerations

1. **Decision Support Prototype:** This AI module is designed for preliminary triage, screening prioritization, and doctor assistance. It is **not** an autonomous diagnostic device.
2. **Pathology Specificity:** The current model performs binary classification (`normal` vs. `abnormal`) and does not localize specific pathology subtypes (e.g., subcapital humerus fracture vs. osteopenia).
3. **Clinical Validation:** Any clinical deployment requires local hospital calibration, radiologist-in-the-loop oversight, and regulatory validation.

---

## 7. Final Recommendation

**Deploy MobileNetV3 Large** with decision threshold **$\tau = 0.33$** as the primary screening model in the Smart Hospital Management System API. ViT-B/16 remains an excellent reference architecture for future multimodal integration (e.g., medical report generation with MedGemma).
