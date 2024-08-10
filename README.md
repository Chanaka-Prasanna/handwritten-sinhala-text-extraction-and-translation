# Sinhala Handwritten Text Recognition and Translation

## Introduction

This project focuses on developing a system to extract Sinhala characters from handwritten images and translate them into English. The project explores three distinct approaches to tackle the challenges of text recognition and translation.

## Tech Stack

- **Image Pre-processing**: OpenCV
- **Character Recognition**: Tesseract OCR, TensorFlow/Keras
- **Translation**: Google Translate API
- **Programming Language**: Python

## Approaches

### 1. Initial Implementation

- **Tech Stack**: OpenCV, Tesseract OCR, Google Translate API
- **Description**: This approach involved using OpenCV for image pre-processing, Tesseract OCR for character recognition, and Google Translate for translating the recognized text.
- **Outcome**: Successfully translated printed Sinhala text images but struggled with handwritten text due to the complexity of handwritten characters.

### 2. Advanced Recognition Using CNN Architectures

- **Tech Stack**: TensorFlow/Keras (AlexNet, LeNet architectures)
- **Description**: Implemented convolutional neural networks (CNNs) using AlexNet architecture for recognizing handwritten Sinhala characters.
- **Outcome**: The models still produced inaccurate results on handwritten images, possibly due to insufficient training data.

### 3. Fine-Tuning Tesseract OCR

- **Tech Stack**: Tesseract OCR, Python
- **Description**: Attempted to fine-tune Tesseract OCR by preparing training data with TXT, BOX, and TR files. Faced challenges in generating .tr files, which prevented successful fine-tuning.
- **Outcome**: Despite significant effort, the fine-tuning process was incomplete due to difficulties with generating .tr files.
