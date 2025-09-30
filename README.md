# CorneaNet-AI

Welcome to **CorneaNet-AI**, a research-driven repository focused on the application of machine learning to ophthalmology, particularly in corneal disease detection. This project aims to advance automated diagnosis using **machine learning and deep learning techniques** to improve reliability in Ophthalmological AI.

⭐ Please leave a star if you find this project useful!

## Problem Overview
Corneal diseases like keratoconus can lead to vision loss if not detected early, placing a heavy burden on patients and healthcare systems. For 18 months, I’ve poured my energy into **CorneaNet-AI** to develop AI-driven tools that enable faster, more accurate diagnosis, supporting ophthalmologists in delivering timely care. Key achievements include:
- **MS39 Descriptive Statistics**: Extracts ~200 features from corneal data to enhance AI model accuracy for clinical use.
- **Topographical Map Reconstruction**: Converts polar to Cartesian coordinates to create visualizations for CNN-based computer vision tasks in hospitals.
- **Zernike Polynomial Retro-Engineering**: Reconstructs polynomial maps to model corneal shape irregularities, improving diagnostic precision.


## 🚀 Project Overview
- **Domain**: Medical AI, Ophthalmology, Corneal Disease Diagnosis
- **Key Techniques**: Geometry, CatBoost, Convolutional Neural Networks (CNNs), Vision Transformers (ViT)
- **Primary Objective**: Build AI models for early, accurate detection of corneal conditions like keratoconus and prediction of eye metrics
- **Real-World Impact**: Enables AI-assisted diagnostics in clinical settings, potentially saving 70k–120k€ annually for the French Public Health system by reducing manual screening time


## 📌 Current Progress
### ✅ Completed
- **Data Transformation**: Polar-to-Cartesian conversion for generating corneal topography images
- **Model Development**: CatBoost and multi-layer CNNs trained on patient metadata and corneal topography images

### ⏳ In Progress
- **Conformal Prediction**: Integrating uncertainty quantification for trustworthy AI predictions
- **Next Steps**: Improving model generalization by combining tabular metadata with images and fine-tuning for diverse patient populations

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
