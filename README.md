# Furniture-Sketch-Classifier
Classical ML classification of furniture categories from sketch-like edge images

**Author:** Vanya Videva  
**Course:** Machine Learning, SoftUni  
**Date:** May 2026  

---

## Research Question

> Can classical machine learning models recognize furniture categories from sketch-like edge images?

## Project Overview

This project investigates whether classical machine learning  models can recognize furniture categories from sketch-like  edge images. Real furniture images are converted into  sketch-like representations using edge detection. Then  handcrafted visual features such as HOG and PCA-based  representations are extracted and used to train classical  machine learning classifiers.

The project is designed as the first phase of a larger idea:

**Phase 1 - Machine Learning Course:** classify furniture  categories from sketch-like images using classical ML.

**Phase 2 - Deep Learning Course:** use the paired  sketch-real dataset for sketch-to-realistic furniture  generation.

**Phase 3 - Demo App:** allow users to upload or draw  furniture sketches and receive predictions or generated  realistic concepts.


> This repository covers Phase 1 only. Phases 2 and 3 will be developed in subsequent courses.


## Data Sources

**Original source:** Bonn Furniture Styles Dataset - Aggarwal et al. (2018)

- Paper: [arXiv:1812.03570](https://arxiv.org/abs/1812.03570)
- Dataset request: [official page](https://cvml.comp.nus.edu.sg/furniture/index.html)

**Processed dataset:** Kaggle - *link soon*

- 5 categories: beds, chairs, dressers, sofas, tables
- 500 images per class for the first pipeline prototype
- 1000 images per class for the baseline version
- The first prototype uses Canny edge detection. Sobel and Laplacian sketches are planned for later comparison.

> Original images obtained via official author request for non-commercial educational use only.  
> Sketches generated as part of this project.

See [References](#references) for full citation.


## Repository Structure

- `notebooks/01_data_exploration.ipynb`  
  Dataset loading, validation, class distribution, and sample visualization

- `notebooks/02_sketch_generation.ipynb`  
  Edge detection comparison - Canny, Sobel, and Laplacian sketch generation methods

- `notebooks/03_feature_extraction.ipynb`  
  HOG and PCA-based feature extraction and visual feature analysis

- `notebooks/04_classification.ipynb`  
  SVM, Random Forest, KNN, and Logistic Regression training and comparison

- `notebooks/05_analysis_results.ipynb`  
  Model evaluation, confusion matrices, and error analysis

- `notebooks/06_conclusions_future_work.ipynb`  
  Project synthesis, limitations, and roadmap to Phase 2 and Phase 3

- `data/raw/`  
  Original Bonn dataset images (not tracked)

- `data/sketches/`  
  Generated edge images (not tracked)

- `src/`  
  Reusable Python modules for sketch generation, feature extraction, and utilities


## Requirements

- Python 3.x
- OpenCV - edge detection and image processing
- scikit-learn - classical ML classifiers
- NumPy / Pandas - data handling
- Matplotlib - visualization
- scikit-image - HOG feature extraction


## References

Aggarwal, D., Valiyev, E., Sener, F., & Yao, A. (2018).Learning Style Compatibility for Furniture.
*German Conference on Pattern Recognition*, 552-566. Springer. arXiv:1812.03570

```bibtex
@inproceedings{aggarwal2018learning,
  title={Learning Style Compatibility for Furniture},
  author={Aggarwal, Divyansh and Valiyev, Elchin and 
          Sener, Fadime and Yao, Angela},
  booktitle={German Conference on Pattern Recognition},
  pages={552--566},
  year={2018},
  organization={Springer}
}
```


## Final Cleanup Checklist

Before submission:

- [ ] Verify all notebooks run locally.
- [ ] Remove unnecessary debug outputs.
- [ ] Keep key plots and result tables.
- [ ] Check that raw data and generated sketches are not tracked.
- [ ] Confirm `outputs/figures/` contains important project figures.
- [ ] Update README with final results.
- [ ] Check GitHub notebook rendering or provide note if preview fails.

