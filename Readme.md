Concrete Crack Classification using Transfer Learning

Project Overview

There are several types of common concrete cracks namely hairline cracks which
usually develop in concrete foundation as the concrete cures, shrinkage cracks which
occur while the concrete is curing, settlement cracks which happen when part of
concrete sinks or when the ground underneath the slab isn’t compacted properly as
well as structural cracks which form due to incorrect design.
Concrete cracks may endanger the safety and durability of a building if not being
identified quickly and left untreated

This project aims to classify concrete surfaces as "Positive" (Cracked) or "Negative" (No Crack) using a deep learning model based on Transfer Learning with MobileNetV2. The dataset consists of images of concrete surfaces categorized into Positive (cracked) and Negative (not cracked) classes.

Dataset

Link to the dataset:
https://data.mendeley.com/datasets/5y9wdsg2zt/2

The dataset is structured as follows:

Concrete Crack Images for Classification/
├── Positive/  # Images of cracked concrete
├── Negative/  # Images of non-cracked concrete

Dependencies

Ensure you have the following Python packages installed:

pip install tensorflow opencv-python numpy matplotlib

How to Run the Project

1. Load the Dataset

The dataset is loaded using image_dataset_from_directory with an 80-20 split for training and validation.

2. Model Architecture

MobileNetV2 is used as the feature extractor.

Data augmentation (random flipping and rotation) is applied.

A custom classifier is added on top of the pre-trained model.

3. Training Process

First Stage: Train only the classifier layers while freezing MobileNetV2.

Second Stage: Fine-tune the last layers of MobileNetV2.

TensorBoard is used for monitoring training progress.

4. Save the Trained Model

After training, the model is saved in the saved_models directory:

model.save("saved_models/concrete_crack_classifier.keras")

5. Model Prediction

To classify a new image:

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("saved_models/concrete_crack_classifier.keras")

# Load and preprocess an image
img = cv2.imread("path/to/image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (160, 160))
img = np.expand_dims(img, axis=0) / 255.0  # Normalize

# Predict
prediction = model.predict(img)
predicted_class = ["Negative", "Positive"][np.argmax(prediction)]
print("Predicted Class:", predicted_class)

Tensorboard Epoch Accuracy:
![alt text](<Epoch Accuracy.png>)



Tensorboard Epoch Loss:
![alt text](<Epoch Loss.png>)


Results and Performance

The model achieves high accuracy (>85%) and an F1-score >0.7.

Transfer learning significantly improves classification performance.

Future Improvements

Increase dataset size for better generalization.

Experiment with different architectures (ResNet, EfficientNet, etc.).

Deploy as a web app using Flask or FastAPI.

Author: Ahnaf

This project is part of the Capstone 3 coursework.

