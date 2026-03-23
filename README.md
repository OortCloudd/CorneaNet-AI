# CorneaNet-AI
 
Reverse engineering the MS-39 corneal topographer and building ML pipelines for corneal disease diagnostics at Quinze-Vingts Hospital, Paris.
 
## What This Actually Does
 
The MS-39 exports corneal measurements as polar-coordinate CSV files in a semi-proprietary format. This repository contains the engineering work required to turn those raw exports into clinically useful ML predictions:
 
- **Polar-to-Cartesian coordinate reconstruction** of corneal topography maps (256×256 Cartesian matrices from raw polar grids, with cubic interpolation for missing data)
- **Zernike polynomial decomposition** up to 7th order, with Best Fit Sphere centering and optical path difference correction (refractive index 1.376). Built from scratch after reverse engineering how the MS-39 computes its internal wavefront representation
- **Statistical feature extraction** producing ~4,000 features per patient across topographic, tomographic, and biomechanical data sources
- **Topographic map reconstruction** for CNN training (11 corneal map types per eye)
 
## Models and Results
 
| Model | Task | Performance |
|-------|------|-------------|
| XGBoost | Surgical candidate selection for intracorneal ring segment implantation | Perfect discrimination *(published in Cornea, 2025 — co-first author)* |
| TabPFN | Classification of seven corneal conditions from biomechanical data | 88.7% accuracy *(published in British Journal of Ophthalmology, 2025)* |
| CatBoost | Prediction of postoperative visual acuity | 0.7 lines MAE |
 
## Publications
 
- Perez E\*, Louissi N\*, et al. "Machine Learning Model for Predicting Visual Acuity Improvement After Intrastromal Corneal Ring Surgery in Patients With Keratoconus." *Cornea*, 2025. [link]
- Borderie VM, Georgeon C, Louissi N, et al. "CorvisST biomechanical indices in the diagnosis of corneal stromal and endothelial disorders." *British Journal of Ophthalmology*, 2025. [link]
 
## What's Next
 
Investigating SOTA ray tracing methods applied to corneal surface models for early keratoconus detection, aiming to improve on the current clinical gold standard (Belin/Ambrosio Enhanced Ectasia Display).
