import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
import pandas as pd
import librosa
import soundfile as sf
import time

class Cutter():
    def __init__(self, dataset_path, output_dataset_path):
        self.dataset_path = dataset_path
        self.output_dataset_path = output_dataset_path

    def cut_audio_file(self, filename, sample_rate=8000, sample_duration=5):
        print(f"Cutting {filename}")
        t0 = time.time()
        sample_length = int(sample_rate * sample_duration)
        
        sample, source_sample_rate = torchaudio.load(os.path.join(self.dataset_path, filename))
        sample = torchaudio.transforms.Resample(orig_freq=source_sample_rate, new_freq=sample_rate)(sample)
        samples = torch.split(sample, sample_length, 1)
        samples = samples[:-1]
        for k, new_sample in enumerate(samples):
            # May try to perform VAD here
            output_file_name = filename.replace('.wav', f"_{k}.wav")
            output_file_path = os.path.join(self.output_dataset_path, output_file_name)
            torchaudio.save(output_file_path, new_sample, sample_rate)
        print(f"Cut {filename} into {len(samples)} samples in {time.time()-t0} seconds")

    def run_batch(self, batch):
        t0 = time.time()
        # with Pool(processes = len(batch)) as pool:
        #     pool.map(self.cut_audio_file, batch)
        for filename in batch:
            self.cut_audio_file(filename)
        print(f"Batch processed in {time.time()-t0} seconds.")

def run(dataset_path, output_dataset_path, batch_length=4):
    filenames = os.listdir(dataset_path)
    os.makedirs(os.path.join(output_dataset_path), exist_ok=True)

    cutter = Cutter(dataset_path, output_dataset_path)
    batches = []
    for index in range(0, len(filenames), batch_length):
        batch = filenames[index: min(index+batch_length, len(filenames))]
        batches.append(batch)

    for batch in batches:
        cutter.run_batch(batch)

if __name__ == "__main__":
    dataset_path = os.path.join('dataset')
    output_dataset_path = os.path.join('dataset_cut')

    run(dataset_path, output_dataset_path)
