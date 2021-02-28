# Presentation
Deep Learning project: detect depression from speech based on DAIC-WOZ
Trained with Pytorch



# Descriptions of files
- vad.py: transforms a folder of .wav files into a folder of cleaner .wav data
- cut_audios.py: cuts a folder of large .wav files into 5s .wav files
- load_data.py: generates a custom DataLoader that yields batches of melspectrograms from 5s .wav files
- aggregator.py: generates a custom DataSet that yields batches of all the melspectrograms corresponding to 1 participant, participant by participant. Manages evaluation
- models.py: model hub, yields models and lets us study their structure
- depression.py: fetches data and trains a model. Evaluates and saves it after each epoch

# Installation
Trained on Cuda 11.0 using Pytorch on Python 3.8.
Requires conda and a lot of libraries.

# Running
Edit the directories of your datasets in the resptive files if they aren't correct.
'''
conda activate traumavoice_dl
D:
cd Deep Learning
python vad.py
python cut_audios.py
python depression.py
'''

# Walkthrough of the construction
## Pre-processing
We have ~20m sound files. We apply a rather aggressive VAD, dropping them down to 10m. We then cut them into 5s samples, transform them into spectrograms, convert them to melspectrograms, and then resize things to the format desired by the model.
ResNet models handle the batchnormalization themselves.

## Model handling - Transfer learning onto ResNet
We attempt to perform transfer learning using the ResNet models.
This model is meant as a 1000-class classifier on RGB images.
We start by converting our grayscale spectrograms to RGB (by copying them into all 3 dimensions).
We then import the resnet model.
We apply different learning rates based on how far they are in the model.
The last classifying layer is a 1000-class layer with sigmoid classifiers. We replace that one with our own layers : a few 256-neurons layers and a 1-neuron layer, with some dropout and a LeakyReLus.
We then train the model.
After some training, we save the model and the train-test split it used (so that we can perform testing on its the correct test set later later).

Model weight initialization is left as default, which for pytorch is the Xavier initialization.
The dataset isn't balanced, we have 11213 positive and 24847 negative samples for 5s samples. However, this is not as much of an issue for the regression task, and the BCEWithLogitsLoss can handle this uneven distribution by giving more weight to the positive samples. We're starting with a classification task because it is slightly easier to work with at the beginning. We can however expect slightly higher performances with regression, as all of the information in the classification task is still known in the regression task. Indeed, one person is deemed "depressed" when their depression score is at least 10.

We keep 25% (~9010) samples as test and 75% (~27030) as train.

## Model training
First approach:
- With a homemade very simple CNN, doesn't yield anything
- The model structure is so basic there was no way it would work
- But it runs

Second approach:
- Transfer learning: using pretrained weights and freezing 
- Using resnet34
- No good, training doesn't seem to improve performances whatsoever

Third approach:
- Same, using the Resnet34 architecture again, but without having them pretrained: we train them ourselves
- We reach 50% accuracy (versus a 31% baseline then for random! -- Or a 69% baseline if it only answered "not depressed") after 3 epochs.

At this point, the strategy of keeping all the data with heavier weight towards positive samples was dropped. Now we balance the dataset ahead of time.
We this new balance, the complete dataset is effectively 20k samples, split between 15k in train and 5k in test.

We have a model that seems to be able to train. Although we don't expect much in terms of performances at this point, it is enough to build the complete pipeline.

## Aggregation
We create a new tool: an aggregator. This tool builds its own dataset, very similar to the former one, but with a few differences. It lets us load all of the files that come from a given person as a single batch. This allows us to retrieve the predictions of our model on every of those samples. The goal is then to aggregate that data, using statistical tools to compute the original depression score / status.

Ideally, we'd have a cleanly split repartition : all of the depressives would be above a given horizontal line (the 50% threshold would be the ideal case), while all of the not depressive ones would be under it. 
This isn't the case yet, and we can clearly see that with this model and at this point, no linear classifier will be able to split the data apart between depressed and not depressed. However, the the model seems to have a vague idea who is depressed and who isn't.

## Improvements
We may now improve everything a bit:
- Spectrograms are replaced with MelSpectrograms, which are basically spectrograms with a rescaled amplitude scale, designed to better fit the human ear and voice.
- We adapt things to be able to turn it into a regression task
- We introduce Dropout on the last layers (the ones added after transferring from the original model)
- We switch from Resnet34 to Resnet50

This depression recognition is essentially a form of pattern recognition here: recognizing the deformation caused by depression on a human's voice. Hence, we can assume that ResNet has a hope of reaching decent accuracies. However, Resnets are trained to recognize real-world shapes : ears, wheels, etc, whereas we must recognize some spectrogram-world shapes here. This is quite different.

- We attempt to use the smaller SqueezeNet, but with no convincing performances
- We use AdamW as our new optimizer of choice because it looks fancy

## Additional pre processing: VAD
We add one step in our data processing : before cutting the data, we pass it through a VAD (Voice Activity Detection) system. This attempts to remove parts where no speech is detected. This removes silence, very noisy parts, and natural sounds (doors opening, microphones being setup, etc.). Moreover, as the interviewer's voice is quite low in those samples, it usually gets trimmed out.
We use WebRTCVAD, an open-source python library to perform this part, using 30-ms frames with some smoothing. WebRTCVAD lets us chose an "aggresivity" from 0 to 3. Simply put, chosing 0 filters silence, but struggles to remove the interviewer's speech, whereas chosing 3 may remove some of the patient's speech. We chose 2, otherwise the interviewer's questions would be too prevalent in the audio. The interviewer can still be heard, but never longer than a few seconds at a time. After a few listens, about 90% of the patient's original speech is kept, though the end of phrases may be cut abruptly. 
Overall, this step removes about 50% of our dataset, most of which was only just noise. This is not negligible, but is definitely necessary.
This step may cut phrases abrutply. This renders some of the features used by feature-based algorithms inefficient. Most notably, rate of speech may be correlated with depression, which the model won't be able to use anyway.

We replace the test routine by the aggregation, which has much more sense here.
The area under the ROC (Accuracy = f(recall)) curve is used instead of the accuracy as the new metric.

# TODO
- Parse args from command line instructions ?
- Merge aggregator(dataset) & load_data
- Merge aggregator(eval) & depression
- Compare with same pipeline but regression
- Build samples with windows that cover eachother a bit
- 1D Convolutions
- Compare freeze+finetune vs different lrs
- Data augmentation
- LSTMs
- Different aggregation weights or methods