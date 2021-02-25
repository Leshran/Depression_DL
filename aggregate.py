import load_data
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torchaudio
import torchvision
import load_data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import models
import time
import numpy as np
import sklearn

def display_predictions(predictions, labels, goal):
    # predictions is a list of tensors
    # labels is a list of tensors
    # Both have the same size
    predictions = np.array(predictions)
    labels = np.array(labels)
    if goal == "classification":
        xaxis = np.array(range(len(labels)))
        colors = ['red' if y else 'blue' for y in labels]
        red_patch = mpatches.Patch(color='red', label='Depressed')
        blue_patch = mpatches.Patch(color='blue', label='Not Depressed')
        plt.legend(handles=[red_patch, blue_patch])
        plt.xlabel("Samples")
    else:
        xaxis = labels
        plt.xlabel("Actual depression score")
        patch1 = mpatches.Patch(color=plt.cm.viridis(0), label='Great prediction')
        patch2 = mpatches.Patch(color=plt.cm.viridis(75), label='Decent prediction')
        patch3 = mpatches.Patch(color=plt.cm.viridis(150), label='Mediocre prediction')
        plt.legend(handles=[patch1, patch2, patch3])
        colors = abs(predictions - labels)
    plt.scatter(xaxis, predictions, c=colors)
    plt.ylabel("Predicted value")
    plt.show()

def evaluate_predictions(predictions, labels, display):
    """
    # predictions are a 1D-array containing probabilities of belonging to each class
    # Trains a linear classifier on 1-D, which basically just finds a threshold above which samples are considered "depressed", and under which they aren't
    this function will draw the ROC curve for the predictions and labels and return the area under curve
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, predictions)
    auc = sklearn.metrics.auc(fpr, tpr)
    if display:
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0,1], [0,1], '-', label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curve - AUC: {auc}")
        plt.show()
    return auc

class AggregatorDataset(Dataset):
    def __init__(self, filenames, dataset_path="dataset_vad", target_df_path="targets.csv", goal="classification", sample_duration=5):
        self.dataset_path = dataset_path
        self.target_df = pd.read_csv(target_df_path)
        self.filenames = filenames
        self.goal = goal
        self.sample_duration = sample_duration

        file_path = os.path.join(self.dataset_path, self.filenames[0]) # Load one file to get sample rate
        _, sample_rate = torchaudio.load(file_path) 
        self.source_sample_rate = sample_rate
        self.sample_rate = 8000
        self.spectrogram_shape = (224, 224)

    def transformer(self, sample, rgb = True, batch_size=16):
        sample_length = int(self.sample_rate * self.sample_duration)
        full_batch = torch.split(sample, sample_length, 1)
        full_batch = torch.stack(full_batch[:-1]) # Removing the trailing sample, whose size could different from the rest. Sorry!

        transforms = torch.nn.Sequential(
            torchaudio.transforms.Resample(orig_freq=self.source_sample_rate, new_freq=self.sample_rate),
            torchaudio.transforms.MelSpectrogram(),
            torchvision.transforms.Resize(self.spectrogram_shape),
        )
        full_batch = transforms(full_batch)
        if rgb: # The spectrograms are grayscale by default
            full_batch = full_batch.repeat(1, 3, 1, 1)
        mini_batches = torch.split(full_batch, batch_size, 0)
        return mini_batches

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.filenames[idx]
        # print(f"Preparing batch corresponding to {filename}")
        file_path = os.path.join(self.dataset_path, filename)
        sample, _ = torchaudio.load(file_path)
        mini_batches = self.transformer(sample)
        label = load_data.get_label(filename, self.target_df, goal=self.goal)
        labels = torch.tensor(label)
        # print(f"Batch shapes: {[batch.shape for batch in mini_batches]}")
        return mini_batches, labels

def evaluate(dataset, model, goal="classification", display = False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model.to(device)
    model.eval()
    # correct = 0
    predictions = []
    labels = []
    with torch.no_grad():
        t0 = time.time()
        for mini_batches, y in dataset: # 1 set of mini_batches corresponds to all the samples from 1 person
            y = y.float()
            y = y.to(device)
            sample_preds = []
            for X in mini_batches:
                X = X.float()
                X = X.to(device)
                out = model(X)
                sample_preds.append(out)
            sample_preds = torch.cat(sample_preds)
            # print(sample_preds, y)
            # print(f"Average sample_preds: {sample_preds.mean():.3f} - real label: {y[0]}")
            predictions.append(sample_preds.mean().item())
            labels.append(y.item())
            sample_preds = sample_preds.round()
    auc = evaluate_predictions(predictions, labels, display)
    print(f"Evaluation done. Auc = {auc} - {time.time()-t0:.4f}s")
    if display:
        display_predictions(predictions, labels, goal)
    return auc

    # correct += prediction.eq(y).sum().item()
    # taux_classif = 100. * correct / len(dataset)
    # print('Accuracy: {}/{} (tx {:.2f}%, err {:.2f}%)\n'.format(correct, len(dataset), taux_classif, 100.-taux_classif))
    # print(f"Evaluation done in {time.time() - eval_start_time}")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print(f"Running on {device}.")

    goal = "classification"
    # goal = "regression"
    
    dataset_path = os.path.join('dataset')
    target_df_path = os.path.join('targets.csv')
    filenames = os.listdir(dataset_path)
    splits_path = "splits"
    splits_name = "20_2021_02_24_13_56"
    _, test_files = load_data.load_train_test(filenames, splits_path, splits_name)
    dataset = AggregatorDataset(test_files, dataset_path, target_df_path, goal, sample_duration=5)

    models_path = os.path.join('models', 'resnet50')
    model = models.ResNet50(pretrained=True, goal=goal)
    model_name = "20_2021_02_24_15_00"
    model.load_state_dict(torch.load(os.path.join(models_path, model_name)))
    print("Loaded model", model_name)
    ## Aggregate
    evaluate(dataset, model, goal=goal, display=False)

    ## Display spectrogram
    # for batch, y in dataset:
    #     X = batch[0]
    #     print("Batch with", len(X), "files")
    #     tensor_image = X[0]
    #     print("Displaying spectrogram of shape", tensor_image.shape)
    #     plt.imshow(tensor_image.permute(1, 2, 0))
    #     plt.show()
    #     break

    # Display graph
    # Xs = list(np.random.random(10))
    # ys = [0,1,0,1,1,0,1,0,1,0]
    # print("Displaying", Xs, ys)
    # display_predictions(Xs, ys, "regression")

    # Evaluate predictions
    # Xs = list(np.random.random(10))
    # ys = [0,1,0,1,1,0,1,0,1,0]
    # print("Evaluating", Xs, ys)
    # evaluate_predictions(Xs, ys, None)