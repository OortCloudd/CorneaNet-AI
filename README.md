# CorneaNet-AI

Welcome to **CorneaNet-AI**, a research-driven repository focused on the application of machine learning to ophthalmology, particularly in corneal disease detection. This project aims to advance automated diagnosis using **machine learning and deep learning techniques** to improve reliability in Ophtalmological AI.

⭐ Please leave a star if you find this project useful!

## 🚀 Project Overview

- **Domain:** Medical AI, Ophthalmology, Corneal Disease Diagnosis
- **Key Techniques:** Geometry, CatBoost, Convolutional Neural Networks (CNNs), ViT
- **Primary Objective:** Develop robust AI models for early and accurate detection of corneal conditions such as keratoconus and prediction of various eye metrics
- **Real-World Impact:** Enabling AI-assisted diagnosis in clinical settings, saving 70k-120k€ every year for the French Public Health system.

---

## 📌 Current Progress

### ✅ Completed
- **Data Transformation:** Polar to Cartesian transformation to reproduce topography images of the cornea.
- **Model Development:** Catboost and CNN-based architectures trained on patient metadata corneal topography images.

### ⏳ In Progress
- **Conformal Prediction Integration:** Uncertainty quantification for trustworthy AI predictions.
- **Next Steps:** Enhancing generalization across patient populations by combining tabular metadata with images and fine-tuning models

---

## 🛠 Technical Details

### **Data**
- Multi-source corneal imaging datasets
- Preprocessing with feature extraction & normalization

### **Model Architecture**
- CNN-based classifiers trained for **keratoconus vs. normal vs. other conditions**
- Conformal prediction for calibrated uncertainty estimates

### **Performance Metrics**
- **Accuracy, Sensitivity, Specificity**
- **ROC AUC** (*not ideal but necessary for communication with ophthalmologists*)
- **LogLoss, Cross-Entropy**
- **R², MAE, RMSE**
- **Uncertainty-aware AI evaluations**

---

## 📢 Contributing
Contributions are welcome! Feel free to submit issues, feature requests, or pull requests to improve the project.

## 📄 License
This project is licensed under the MIT License.

## 📬 Contact
For any inquiries, feel free to reach out via GitHub issues or email.
