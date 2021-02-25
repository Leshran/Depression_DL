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

## Model handling - Transfer learning onto ResNet
We attempt to perform transfer learning using the ResNet34 model
This model is meant as a 1000-class classifier on 3D images.
We start by converting our grayscale images to RGB (by copying them into all 3 dimensions)
We then import the resnet34 model
We freeze the training of all the parameters of the model
The last classifying layer is a 1000-class layer with sigmoid classifiers. We replace that one with our own layers : a 256-neuron and a 1-neuron layer.
We then perform training on the model.
After some training, we save the model and the train-test split it used (so that we don't perform testing on its train set later)

Model weight initialization is left as default, which for pytorch is the Xavier initialization

The dataset isn't balanced, we have 11213 positive and 24847 negative samples. However, this is not as much of an issue for the regression task, and the BCEWithLogitsLoss can handle this uneven distribution by giving more weight to the positive samples. We're starting with a classification task because it is slightly easier to work with at the beginning. We can however expect slightly higher performances with regression, as all of the information in the classification task is still known in the regression task. Indeed, one person is deemed "depressed" when their depression score is at least 10.

We keep 25% (~9010) samples as test and 75% (~27030) as train.

## Model training

First approach:
    With a homemade very simple CNN, doesn't yield anything
    The model structure is so basic there was no way it would work

Second approach:
    Transfer learning: using pretrained weights and freezing 
    Using resnet34
    No good, training doesn't seem to improve performances whatsoever

Third approach:
    Same, using the Resnet34 architecture again, but without having them pretrained: we train them ourselves
    We reach 50% accuracy (versus a 31% baseline for random! -- Or a 69% baseline if it only answered "not depressed") after 3 epochs.

At this point, the approach of keeping all the data with heavier weight towards positive samples was dropped. Now we balance the dataset ahead of time.
We this new balance, the complete dataset is effectively 22426 samples, split between 16819 in train and 5607 as test.

We have a model that seems to be able to train. Although we don't expect much in terms of performances at this point, it is enough to build the complete pipeline.


## Aggregation
We create a new tool: an aggregator. This tool builds its own dataset, very similar to the former one, but with a few differences. It lets us load all of the files that come from a given person as a single batch. This allows us to retrieve the predictions of our model on every of those samples. The goal is then to aggregate that data, using statistical tools to compute the original depression score / status.

<classification_resnet_1.png>
We get this graph after aggregation. Each dot corresponds to one participant, whose depression status is represented with colors. The y-axis represents the average probability of being depressed the model gave each of the 5s-samples taken from the original file.
Ideally, we'd have a cleanly split repartition : all of the red dots would be above a given horizontal line (the 50% threshold would be the ideal case), while all of the blue ones would be under it. 
This isn't the case yet, and we can clearly see that with this model and at this point, no linear classifier will be able to split the data apart between depressed and not depressed. However, the graph still shows that most of the red dots are in the upper part of the graph : the model seems to have an idea who is depressed and who isn't.
Note that this evaluation was run on the whole dataset, some of which was used as train samples. OOPPS

## Improvements
We may now improve everything a bit:
- Spectrograms are replaced with MelSpectrograms, which are basically spectrograms with a rescaled amplitude scale, designed to better fit the human ear and voice.
- We switch to a regression task
- We introduce Dropout on the last layers (the ones added after transferring from the original model)
- We switch from Resnet34 to Resnet50

This depression recognition is essentially a form of pattern recognition here: recognizing the deformation caused by depression on a human's voice. Hence, we can assume that ResNet has a hope of reaching decent accuracies. However, Resnets are trained to recognize real-world shapes : ears, wheels, etc, whereas we must recognize some spectrogram-world shapes here. This is quite different.

- We attempt to use the smaller SqueezeNet, but with no convincing performances
- We use AdamW as our new optimizer of choice

## Additional pre processing: VAD
We add one step in our data processing : before cutting the data, we pass it through a VAD (Voice Activity Detection) system. This attempts to remove parts where no speech is detected. This removes silence, very noisy parts, and natural sounds (doors opening, microphones being setup, etc.). Moreover, as the interviewer's voice is quite low in those samples, it usually gets trimmed out.
We use WebRTCVAD, an open-source python library to perform this part, using 30-ms frames with some smoothing. WebRTCVAD lets us chose an "aggresivity" from 0 to 3. Simply put, chosing 0 filters silence, but struggles to remove the interviewer's speech, whereas chosing 3 may remove some of the patient's speech. We chose 2, otherwise the interviewer's questions would be too prevalent in the audio. The interviewer can still be heard, but never longer than a few seconds at a time. After a few listens, about 90% of the patient's original speech is kept, though the end of phrases may be cut abruptly. 
Overall, this step removes about 50% of our dataset, most of which was only just noise. This is not negligible, but is definitely necessary.
This step may cut phrases abrutply. This renders some of the features used by feature-based algorithms inefficient. Most notably, rate of speech may be correlated with depression, which the model won't be able to use anyway.

We replace the test routine by the aggregation, which has much more sense here.
The area under the ROC (Accuracy = f(recall)) curve is used instead of the accuracy as the new metric.
# TODO
- Retrain only last n layers
- Drop classifying layer
- Finetune les autres couches avec une régularization qui les force à pas trop bouger

- Remove the test loader which is useless, have load_data create the aggregator

- Data augmentation
- LSTMs
- Different aggregation weights
- Fine-tuning
- Handle train test with a class to handle the paths and remember them (necessary for full pipeline)
- Early stopping routine


# Upgrades
Cut samples so that they cover each other (ie 5s, but the last 2s are also in the beginning of the next sample)
Convolution sur 1 seule dimension ? (mais il faut alors des LSTMs)
Other, more S.O.T.A methods: LSTMs, Transformers
Replace the aggregation phase with an LSTM ?