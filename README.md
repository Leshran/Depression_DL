# Presentation
Deep Learning project
Trained with Pytorch

# Versions
Trained on Cuda 11.0
Using Pytorch
Python 3.8

# Installation
'''
conda activate traumavoice_dl
D:
cd D:\Deep Learning
'''

# Notes
First, data loading and processing using pytorch custom dataloaders, datasets and transformers for efficiency
First, attempted a basic CNN model, didn't perform well
Then, converted to RGB to use ResNet

## Pre-processing
Todo

## Transfer learning onto ResNet
We attempt to perform transfer learning using the ResNet34 model
This model is meant as a 1000-class classifier on 3D images.
We start by converting our grayscale images to RGB (by copying them into all 3 dimensions)
We then import the resnet34 model
We freeze the training of all the parameters of the model
The last classifying layer is a 1000-class layer with sigmoid classifiers. We replace that one with our own layers : a 256-neuron and a 1-neuron layer.
We then perform training on the model 

Model weight initialization is left as default, which for pytorch is the Xavier initialization

The dataset isn't balanced, we have 11213 positive and 24847 negative samples. However, this is not as much of an issue for the regression task, and the BCEWithLogitsLoss can handle this uneven distribution by giving more weight to the positive samples.
We keep 25% (9010) samples as test and 75% (27030) as train.

First approach:
    With a homemade very simple CNN, doesn't yield anything
    The model structure is so basic there was no way it would work

Second approach:
    Transfer learning: using pretrained weights and freezing 
    Using resnet34
    No good, training doesn't seem to improve performances whatsoever

Third approach:
    Same, using the Resnet34 architecture again, but without having them pretrained: we train them ourselves
    50% accuracy (versus a 31% baseline for random!) after 3 epochs

At this point, the approach of keeping all the data with heavier weight towards positive samples was dropped. Now we balance the dataset ahead of time 