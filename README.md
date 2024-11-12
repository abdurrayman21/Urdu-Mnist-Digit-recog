# Urdu MNIST Recognition

This project implements a machine learning model to recognize and classify handwritten Urdu numerals (0–9) using an Urdu adaptation of the popular MNIST dataset. It aims to support optical character recognition (OCR) applications for Urdu digits, contributing to Urdu language processing in machine learning.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [License](#license)

## Overview
The goal of this project is to build a robust Urdu numeral recognition model using deep learning techniques. The repository contains code for data preprocessing, model training, and evaluation. By the end, the model will be able to accurately recognize handwritten Urdu digits, providing insights into OCR tasks for the Urdu language.

## Dataset
This project uses an Urdu version of the MNIST dataset, containing grayscale images of handwritten digits (0–9) in Urdu script.

- **Image Size**: 28x28 pixels, grayscale
- **Classes**: 10 (digits 0–9 in Urdu)

The dataset can be downloaded from [Link to Dataset Source] (update this link as needed).

## Project Structure

## Getting Started

## Model Architechure
The model is based on a Convolutional Neural Network (CNN), optimized for image classification tasks. Key layers include:

Convolutional Layers: Extracts image features with filters.
Pooling Layers: Reduces dimensionality, capturing essential features.
Fully Connected Layers: Combines extracted features to make predictions.

## Training and Evaluation
Open the notebooks/ folder and run Urdu_MNIST_Training.ipynb to preprocess data, train the model, and evaluate results.
1 - Training parameters, such as batch size and number of epochs, can be adjusted in the notebook.
2 - The model is evaluated using metrics including accuracy, precision, recall, and F1 score. A confusion matrix is also generated to visualize classification performance.

### Prerequisites
- Python 3.7+
- Jupyter Notebook (optional)
- TensorFlow / Keras
- Numpy, Pandas, Matplotlib, Seaborn
  
## License 
This project is licensed under the MIT License
