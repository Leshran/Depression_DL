import warnings
warnings.simplefilter("ignore", UserWarning) # Pytorch + multiprocessing means a lot of warnings
import torch
import torchaudio
import torchvision
from torchvision.transforms import Lambda
import os
import time
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import multiprocessing
from functools import partial
import numpy as np
import json

'''
TODO:
Make two types of loaders, one for regression and one for classification
'''

def check_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    return torch.cuda.is_available()

def get_label(filename, target_df, goal="classification"):
    # Returns label using the DAIC_WOZ structure
    Participant_ID = int(filename.split('_')[0])
    if goal == "classification":
        labels = (target_df[target_df["Participant_ID"]==Participant_ID]["PHQ8_Binary"]).values # Classification
    elif goal == "regression":
        labels = (target_df[target_df["Participant_ID"]==Participant_ID]["PHQ8_Score"]).values # Regression
    return labels

def balance_classes(filenames, target_df):
    # Currently unused
    ptsd = []
    no_ptsd = []
    for filename in filenames:
        label = get_label(filename, target_df)
        if not label:
            no_ptsd.append(filename)
        else:
            ptsd.append(filename)
    ptsd, no_ptsd = np.array(ptsd), np.array(no_ptsd)
    # print(f"Before balancing, {len(ptsd)} files with PTSD, {len(no_ptsd)} files without PTSD")
    if len(ptsd) > len(no_ptsd):
        ptsd = np.random.choice(ptsd, len(no_ptsd))
    else:
        no_ptsd = np.random.choice(no_ptsd, len(ptsd))
    # print(f"After balancing, {len(ptsd)} files with PTSD, {len(no_ptsd)} files without PTSD")
    return list(ptsd) + list(no_ptsd)

def compute_ptsd_rate(filenames, target_df, goal="classification"):  
    ptsd = 0
    no_ptsd = 0
    with multiprocessing.Pool(4) as p:
        get_label_partial = partial(get_label, target_df=target_df)
        labels = p.map(get_label_partial, filenames)
    ptsd = int(sum(labels))
    no_ptsd = len(filenames) - ptsd
    print(f"{ptsd} positive samples and {no_ptsd} negative samples.")
    return ptsd / no_ptsd

def save_train_test(train, test, splits_path, splits_name):
    if splits_name is None:
        splits_name = str(np.random.random())
    train_path = os.path.join(splits_path, splits_name + "_train.txt")
    test_path = os.path.join(splits_path, splits_name + "_test.txt")
    with open(train_path, 'w') as trainfile:
        json.dump(train, trainfile)
    with open(test_path, 'w') as testfile:
        json.dump(test, testfile)
    print(f"Train split saved to {train_path} ; train split saved to {test_path}.")
    return train_path, test_path

def load_train_test(filenames, splits_path, splits_name):
    train_path = os.path.join(splits_path, splits_name + "_train.txt")
    test_path = os.path.join(splits_path, splits_name + "_test.txt")
    with open(train_path, 'r') as trainfile: # Load Participant_IDs for train people
        train = json.load(trainfile)
    with open(test_path, 'r') as testfile:
        test = json.load(testfile)
    train, test = set(train), set(test) # Hashing to speed up lookup time
    train_files, test_files = [], []
    for filename in filenames: # Load all train files and test files
        source_file = filename.split('_')[0]
        if source_file in train:
            train_files.append(filename)
        else:
            test_files.append(filename)
    return train_files, test_files

def train_split(filenames, test_size=0.25, splits_path="splits", splits_name=None):
    source_files = list(set([filename.split('_')[0] for filename in filenames]))
    train, test = train_test_split(source_files, test_size=0.25)
    train = set(train) # Hashing to speed up lookup time
    test = set(test)
    train_files = []
    test_files = []
    for filename in filenames:
        source_file = filename.split('_')[0]
        if source_file in train:
            train_files.append(filename)
        else:
            test_files.append(filename)
    train, test = list(train), list(test)
    train_path, test_path = save_train_test(train, test, splits_path, splits_name)
    return train_files, test_files

class DaicWOZDataset(Dataset):
    def __init__(self, filenames, dataset_path, target_df, goal="classification"):
        self.dataset_path = dataset_path
        self.target_df = target_df
        self.filenames = filenames
        self.goal = goal

        file_path = os.path.join(self.dataset_path, self.filenames[0]) # Load one file to get sample rate
        _, sample_rate = torchaudio.load(file_path) 
        self.source_sample_rate = sample_rate
        self.sample_rate = 8000
        self.spectrogram_shape = (224, 224)

    def transformer(self, sample, rgb = True):
        transforms = torch.nn.Sequential(
            torchaudio.transforms.Resample(orig_freq=self.source_sample_rate, new_freq=self.sample_rate),
            torchaudio.transforms.MelSpectrogram(),
            torchvision.transforms.Resize(self.spectrogram_shape),
        )
        sample = transforms(sample)
        if rgb: # The spectrograms are grayscale by default
            sample = sample.repeat(3,1,1)
        return sample

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.filenames[idx]
        file_path = os.path.join(self.dataset_path, filename)
        sample, _ = torchaudio.load(file_path)
        sample = self.transformer(sample)
        labels = get_label(filename, self.target_df, goal=self.goal)
        return sample, labels

def get_data(dataset_path='dataset_cut', target_df_path='targets.csv', batch_size=16, shuffle=True, num_workers=0, splits_path='splits', splits_name=None, goal="classification"):
    filenames = os.listdir(dataset_path)
    target_df = pd.read_csv(target_df_path)
    filenames = balance_classes(filenames, target_df)
    # ptsd_rate = compute_ptsd_rate(filenames, target_df)
    train_files, test_files = train_split(filenames,  splits_path=splits_path, splits_name=splits_name)

    train_dataset = DaicWOZDataset(train_files, dataset_path, target_df, goal=goal)
    test_dataset = DaicWOZDataset(test_files, dataset_path, target_df, goal=goal)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_loader, test_loader

if __name__ == "__main__":
    dataset_path = os.path.join('dataset_cut')
    target_df_path = os.path.join('targets.csv')
    train_loader, test_loader = get_data()

    # Plot first spectrogram
    for X, y in train_loader:
        tensor_image = X[0]
        print("Displaying spectrogram of shape", tensor_image.shape)
        plt.imshow(tensor_image.permute(1, 2, 0))
        plt.show()
        break

    # daicWOZDataset = DaicWOZDataset(dataset_path, target_df_path)
    # data_loader = torch.utils.data.DataLoader(daicWOZDataset, batch_size=16, shuffle=True)