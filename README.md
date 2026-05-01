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
[paste the folder structure above]

## How to Run
```bash
git clone https://github.com/yourusername/satellite-image-classifier
cd satellite-image-classifier
pip install -r requirements.txt
# Download EuroSAT dataset from: https://www.kaggle.com/datasets/apollo2506/eurosat-dataset
# Place in data/ folder
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Tech Stack
Python · TensorFlow/Keras · EfficientNetB0 · OpenCV · scikit-learn · Matplotlib

## Author
[Your Name] — BE CSE, [College], 6th Semester