# Emotion Detection Using Deep Learning Architecture

**Author**: Shishir Rijal  
**LinkedIn**: [Shishir Rijal](https://www.linkedin.com/in/ShishirRijal/)  


## Description
This project involves training a facial emotion detection model from scratch using a **Convolutional Neural Network (CNN)** on the **Kaggle FER-2013 Dataset**. The task is to categorize facial expressions into one of seven emotion categories.

The project explores several approaches, including:
1. A custom CNN model built from scratch.
2. A custom CNN model with data augmentation to improve generalization.
3. Transfer learning using **VGG16**.
4. Transfer learning using **ResNet50**.

## Dataset: FER-2013

The **FER-2013** dataset consists of 48x48 pixel grayscale images of faces. These images have been automatically registered so that the face is centered and occupies approximately the same space in each image. The task is to classify each face based on the displayed emotion.

- **Dataset link**: [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- **Image dimensions**: 48x48 pixels, grayscale
- **Number of emotion categories**: 7
  - 0 = Angry
  - 1 = Disgust
  - 2 = Fear
  - 3 = Happy
  - 4 = Sad
  - 5 = Surprise
  - 6 = Neutral

### Sample Image:
Below is a 48x48 pixel grayscale image example from the dataset:

<img width="710" alt="Screenshot 2024-09-21 at 1 04 16 PM" src="https://github.com/user-attachments/assets/8759b6ad-58a8-4c41-a5df-cad0541e9e91">

## Approach Overview

This project applies four different model architectures to detect emotions from facial expressions. Each approach has its own learning method and complexity, ranging from training a model from scratch to leveraging pre-trained models through transfer learning.

### Model Approaches:


#### 1. Custom CNN From Scratch
- A simple CNN architecture designed and trained from scratch.
- Challenges: No data augmentation, prone to overfitting.

#### 2. Custom CNN With Augmentation
- Same architecture as the Custom CNN, but with image augmentation techniques like rotation, zoom, and horizontal flipping.
- Benefit: Reduces overfitting and improves model generalization.

#### 3. VGG16 Transfer Learning
- A VGG16 model pre-trained on ImageNet is fine-tuned for facial emotion detection.
- Benefit: Transfer learning helps leverage pre-learned features for faster convergence and better accuracy.
#### 4. ResNet50 Transfer Learning
- A ResNet50 model pre-trained on ImageNet is fine-tuned for this task.
- Benefit: Deep residual connections allow training deep networks effectively without vanishing gradients.

## Results
- Custom CNN From Scratch:
    - Train accuracy = 88.36%, Validation accuracy = 60.87%
- Custom CNN With Augmentation:
    - Train accuracy: 57.05%, Validation accuracy: 55.84%
- VGG16 Transfer Learning:
    - Train accuracy: xx.xx%, Validation accuracy: xx.xx%
- ResNet50 Transfer Learning:
    - Train accuracy: 60.18%, Validation accuracy: 60.00%

The transfer learning model ResNet50 performed significantly better compared to models trained from scratch, thanks to the pre-learned features from large-scale datasets.

## Conclusion
This project demonstrates various methods to train emotion detection models using facial expressions. Transfer learning with ResNet50 provides better accuracy compared to custom-built models. Future improvements can include exploring more advanced augmentation techniques or fine-tuning the transfer learning models further.
