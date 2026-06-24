# M2D: Target-Label-Free Artificial Intelligence Framework for Cross-Anatomical RNFL Biomarker Segmentation in Optical Coherence Tomography

This repository contains the core architectural implementation and inference pipeline for the M2D framework. 

## Overview
The M2D framework is designed for automated circumpapillary retinal nerve fiber layer (cpRNFL) delineation across mixed-device clinical cohorts. It utilizes a topological abstraction strategy alongside a Domain Adapter Layer to mitigate hardware-induced domain shifts between varying OCT platforms (e.g., Spectralis and BMizar).

## Repository Structure
- `models/`: Contains the PyTorch implementation of the core M2D architecture (EfficientNet-B3 encoder + UPerHead decoder).
- `inference.py`: Standardized inference script detailing the preprocessing pipeline and evaluation logic.
- `requirements.txt`: Environment dependencies for deterministic reproducibility.
- `weights/`: Directory allocated for pre-trained model parameters.

## Reproducibility & Preprocessing
To ensure complete methodological transparency as detailed in our study:
- All input OCT B-scans are strictly resized to 512x512 pixels.
- Image intensity is normalized to a [0, 1] range prior to network ingestion.
- Random seeds are fixed to guarantee deterministic outputs.

## Status
**Currently, this repository provides the structural codebase for peer review.** 
The remaining training pipelines, cross-validation scripts, and pre-trained model weights are currently being curated and are scheduled for public release upon publication of this work.

## Citation
If you find this framework useful in your research, please consider citing our work:
```bibtex
@article{M2D_Framework,
  title={Target-Label-Free Artificial Intelligence Framework for Cross-Anatomical RNFL Biomarker Segmentation in Optical Coherence Tomography},
  author={Anonymous},
  journal={Under Review},
  year={2026}
}
