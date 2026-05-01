# Satellite Image Classification using Deep Learning

Classifying land use and land cover from satellite imagery using 
transfer learning on the EuroSAT benchmark dataset.

## Motivation
Earth Observation satellites like ISRO's EOS-08 generate massive volumes 
of multispectral imagery. Automated classification of land cover is 
critical for environmental monitoring, disaster response, and urban planning.

## Dataset
- **EuroSAT** — 27,000 labeled Sentinel-2 satellite images
- 10 classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, 
  Industrial, Pasture, PermanentCrop, Residential, River, SeaLake
- Image size: 64×64 pixels, RGB

## Model Architecture
- Base: EfficientNetB0 (pretrained on ImageNet)
- Fine-tuned on EuroSAT
- Additional: Grad-CAM visualizations for explainability

## Results
| Metric | Value |
|--------|-------|
| Test Accuracy | ~95% (updating after training) |
| Model Size | ~20MB |

## Project Structure
satellite-image-classifier/
│
├── data/
│   └── EuroSAT dataset
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation_visualization.ipynb
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   └── utils.py
│
├── results/
│   └── confusion matrix
    └── sample predictions
│
├── requirements.txt
├── README.md
└── .gitignore


## Tech Stack
Python · TensorFlow/Keras · EfficientNetB0 · OpenCV · scikit-learn · Matplotlib

## Author
POORVIKA SRINIVAS — BE CSE, SJB Institute of Technology, 6th Semester
