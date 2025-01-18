# RTML_Assignment_1

This repository contains implementations and experiments comparing the performance of AlexNet with Local Response Normalization (AlexNetWithLRN) and GoogLeNet with auxiliary classifiers (GoogLeNet) on the CIFAR-10 dataset. Both models are trained from scratch, and their pretrained counterparts from torchvision are fine-tuned for comparison.

Project Structure

alexnet_with_lrn.py: Implementation of AlexNet with Local Response Normalization.

googlenet_modified.py: Implementation of GoogLeNet with auxiliary classifiers.

main_train.ipynb: Jupyter notebook for training, evaluating, and comparing the models.

Dataset
We used the CIFAR-10 dataset:

60,000 images (32x32 pixels, 10 classes)

Preprocessing: resized to 128x128, center-cropped to 112x112, normalized using CIFAR-10-specific statistics.
