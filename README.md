# FraudShield: Credit Card Fraud Detection System

FraudShield is a robust credit card fraud detection system developed using two different machine learning approaches:
1. **XGBoost with PyCaret for Analysis**
2. **Deep Neural Networks (DNN)**

The goal of this project is to accurately detect fraudulent credit card transactions using state-of-the-art machine learning algorithms and deep learning models.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Dataset](https://www.kaggle.com/datasets/chitwanmanchanda/fraudulent-transactions-data)
- [Model 1: XGBoost with PyCaret](#model-1-xgboost-with-pycaret)
- [Model 2: Deep Neural Networks](#model-2-deep-neural-networks)
- [Visualizations](#visualizations)
- [Contribution](#contribution)


## Overview
Credit card fraud detection is critical in the financial sector to prevent significant monetary losses and protect customer assets. FraudShield utilizes machine learning and deep learning techniques to enhance the detection of fraudulent transactions. 

## Installation
To run this project, clone the repository and install the required dependencies.

```bash
git clone https://github.com/username/FraudShield.git
cd FraudShield
```

## Model 1: XGBoost with PyCaret

**XGBoost**: Known for its powerful tree-based algorithm, used in combination with PyCaret for automated analysis.
XGBoost is used as the primary classifier for this model. PyCaret is leveraged to streamline the entire machine learning workflow, from preprocessing to model evaluation.

Steps:
- Preprocessing: PyCaret automates data scaling, encoding, and splitting.
- Model Building: XGBoost is trained using PyCaret’s setup and compare functions.
- Analysis: PyCaret provides metrics such as accuracy, precision, recall, F1 score, and AUC for evaluation.

## Model 2: Deep Neural Networks

**Deep Neural Networks**: A custom-built deep learning architecture optimized for fraud detection tasks.
A custom deep learning model was built using TensorFlow/Keras, designed to handle the imbalanced nature of the dataset and optimize fraud detection performance.

Architecture:
- Input Layer: Number of features after scaling.
- Hidden Layers: 128, 64, and 32 neurons with ReLU activation.
- Dropout Layers: Added to prevent overfitting.
- Output Layer: A single neuron with sigmoid activation for binary classification.

## Visualizations

- Confusion Matrix
For both models, confusion matrices are plotted to better understand false positives and false negatives.

- Precision-Recall Curve
Illustrates the tradeoff between precision and recall for different thresholds.

- ROC Curve
The ROC curve is plotted to visualize the model's performance across various decision thresholds.

- Loss and Accuracy Curve (for DNN)
Shows the model's learning progress during training.

## Contribution
Contributions are welcome! If you would like to add new models or improve the current implementation, feel free to fork the project and submit a pull request.


