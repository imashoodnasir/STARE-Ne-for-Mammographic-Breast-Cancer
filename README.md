# STARE-Net: Stage-wise Transformer Architecture with Refinement for Mammographic Breast Cancer Classification

This repository implements **STARE-Net**, an explainable deep learning model for classifying mammographic breast cancer using a multi-stage transformer-based pipeline with semantic refinement and ensemble decision-making.

---

## 🧠 Key Modules

- **preprocess.py**: Outlier removal, imputation, normalization, Gaussian filtering
- **backbone.py**: Feature extraction using pretrained MobileNetV2
- **hmt.py**: Hierarchical Multi-Scale Transformer with cross-scale attention
- **elt.py**: Edge-aware Local Transformer guided by entropy
- **tan.py**: Two-Stage Attention Network to combine global and local features
- **acr.py**: Adaptive Contextual Refinement for semantic enhancement
- **ensemble.py**: Meta-ensemble classifier using WSVM + KNN + WMV
- **train.py**: Model construction, training pipeline, and scheduler
- **explain.py**: Explainability using Grad-CAM, SHAP, and Occlusion Sensitivity

---

## 🗂️ Directory Structure

```
STARE-Net/
├── preprocess.py
├── backbone.py
├── hmt.py
├── elt.py
├── tan.py
├── acr.py
├── ensemble.py
├── train.py
├── explain.py
└── README.md
```

---

## ⚙️ Requirements

```bash
pip install tensorflow opencv-python scikit-learn shap matplotlib tensorflow-addons
```

---

## 🚀 How to Run

1. **Preprocess and prepare data**:
   - Apply normalization, imputation, and filtering.
   - Resize mammograms to 224x224.

2. **Train STARE-Net**:
   ```bash
   python train.py
   ```

3. **Generate Explanations**:
   ```bash
   python explain.py
   ```

---

## 🧪 Model Summary

- Input Size: 224×224 mammogram patches
- Backbone: MobileNetV2
- Transformers: HMT + ELT
- Classifier: Meta-ensemble (WSVM + KNN)
- Explainability: Grad-CAM, SHAP, OSA

---

## 🔓 License

This project is licensed under the MIT License.
