# Image-Based Plant Disease Detection using CNN 🌿🦠

## Overview
This project implements a **Convolutional Neural Network (CNN)-based deep learning model** for detecting plant diseases from leaf images. Using computer vision and deep learning techniques, the model classifies different plant diseases and can assist farmers in early disease diagnosis.

## Features
✅ Uses **Convolutional Neural Networks (CNNs)** for image-based classification  
✅ Trained on a **large dataset of plant leaf images** with multiple disease categories  
✅ Implements **transfer learning** for improved accuracy (optional)  
✅ Provides **real-time disease detection** using OpenCV for webcam-based input  
✅ Easy-to-follow Jupyter Notebook for reproducibility  

## Technologies & Libraries Used
This project utilizes various deep learning and computer vision technologies:
- **Programming Language**: Python  
- **Deep Learning Frameworks**: TensorFlow, Keras  
- **Computer Vision**: OpenCV  
- **Data Manipulation**: Pandas, NumPy  
- **Visualization**: Matplotlib, Seaborn  
- **Model Evaluation**: Scikit-learn (classification report, confusion matrix)  

## Dataset
The PlantVillage dataset was used, which consists of 70,000+ **leaf images** categorized by disease type. Each image is labeled with a corresponding disease or marked as "healthy." The dataset is preprocessed by:
- **Resizing images** to a fixed input size for CNN  
- **Normalizing pixel values** for better model convergence  
- **Splitting into training, validation, and test sets**  

## Model Architecture
The CNN model includes:
- **Convolutional Layers**: Extract features from leaf images  
- **Pooling Layers**: Reduce spatial dimensions and prevent overfitting  
- **Fully Connected Layers**: Make predictions based on extracted features  
- **Activation Functions**: ReLU for feature extraction, Softmax for classification  

### CNN Architectures Used
The following CNN architectures were implemented and evaluated:
- **EfficientNet V2B0**
- **DenseNet121**
- **Xception**

### Ensemble Models
To improve classification accuracy, **ensemble approaches** was implemented using:
- **Ensemble 1**: Xception + EfficientNet
- **Ensemble 2**: DenseNet + EfficientNet
- **Ensemble 3**: DenseNet + Xception
- - **Ensemble 4**: DenseNet + EfficientNet + Xception

