# X-Ray Model Evaluation Report

## Task

Binary classification of MURA v1.1 musculoskeletal X-ray images:

- normal
- abnormal

## Models Compared

| Model | Architecture Type | Notes |
| --- | --- | --- |
| MobileNetV3 Large | CNN | Lightweight baseline |
| ViT-B/16 | Vision Transformer | Transformer baseline |

## Training Configuration

| Setting | Value |
| --- | --- |
| Dataset | MURA v1.1 |
| Image size | 224 x 224 |
| Batch size | 32 |
| Optimizer | AdamW |
| Loss | BCEWithLogitsLoss with positive class weighting |
| MobileNetV3 learning rate | 1e-4 |
| ViT-B/16 learning rate | 1e-5 |

The task document lists `0.0001` as the example learning rate. MobileNetV3 uses this value. ViT-B/16 uses `0.00001` because initial testing with `0.0001` produced unstable or weak validation behavior for the larger transformer model.

## Results

Fill this table from `mobilenet_vs_vit_metrics.csv`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Inference ms/image | Model size MB | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MobileNetV3 |  |  |  |  |  |  |  |  |
| ViT-B/16 |  |  |  |  |  |  |  |  |

## Interpretation

Describe:

- Which model has better predictive performance.
- Which model is faster and smaller.
- Whether the transformer improvement is worth the extra compute cost.
- Any signs of overfitting or underfitting.

## Limitations

- This system is not a final clinical diagnostic tool.
- Performance depends on MURA labels and image quality.
- Validation is required before real hospital use.
- The model predicts normal vs abnormal only, not exact disease type.

## Conclusion

Write the final recommendation after full training.
