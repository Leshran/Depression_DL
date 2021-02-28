import warnings
warnings.simplefilter("ignore", UserWarning) # Pytorch + multiprocessing means a lot of warnings
import load_data
import models
import aggregate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.autograd import Variable
import os
import time
import datetime
import sys 

class ModelManager():
    def __init__(self, model, models_path, device, train_loader, test_loader, aggregator, goal):
        self.model = model
        self.models_path = models_path
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.aggregator = aggregator
        self.goal = goal
        if goal == "classification":
            self.criterion = nn.BCELoss() # Classification - Binary Cross Entropy, converted to the [0, 1] range
        else:
            self.criterion = nn.MSELoss() # Regression - Mean Squared Error
    
    def train(self, epochs, verbose=False):
        self.model.train()
        optimizer = torch.optim.AdamW([
            {'params': self.model.resnet50.parameters(), 'lr': 0.001}, # Low learning rate for the pretrained layers
            {'params': self.model.fc.parameters(), 'lr': 0.01}  # Higher learning rate for the final classifying layers
        ])
        
        epoch_losses = []
        AUCs = []
        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            epoch_start_time = time.time()
            batch_start_time = time.time()
            epoch_samples_count = 0
            for step, (X, y) in enumerate(self.train_loader): # for each training step
                X, y = X.float(), y.float() # Convert as they're stored as doubles
                X, y = Variable(X).to(self.device), Variable(y).to(self.device) # Bring to GPU
                prediction = self.model(X)     # input x and predict based on x
                loss = self.criterion(prediction, y)     # compute loss
                epoch_loss += loss.item() * prediction.shape[0]
                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagate
                optimizer.step()        # apply gradients
                batch_duration = time.time() - batch_start_time
                epoch_duration = time.time() - epoch_start_time
                batch_start_time = time.time()
                epoch_samples_count += len(X) 
                sys.stdout.write(f"\rEpoch {epoch} - ({epoch_samples_count}/{len(self.train_loader.dataset)}) - Loss: {loss.item():.3f} - epoch: {epoch_duration:.3f}s - step: {batch_duration:.3f}s") # Cute inline logging
                if step % 10 == 0 and verbose:
                    print(f"Epoch {epoch} - ({epoch_samples_count}/{len(self.train_loader.dataset)}) - Last prediction: {prediction} vs {y}")
            epoch_time = time.time() - epoch_start_time
            epoch_losses.append(epoch_loss)
            print(f"\nEpoch {epoch} done. Average loss: {(epoch_loss/len(self.train_loader.dataset)):.3f} - {epoch_time:.4f}s")
            if verbose:
                print("Last prediction", prediction)
                print("Last y", y)
            auc = aggregate.evaluate(self.aggregator, self.model) # Evaluate results using the AUC of the ROC for the model on the test data
            AUCs.append(auc)
            models.save_model(self.model, self.models_path, epoch)
        models.save_model(self.model, self.models_path, epoch)
        print("Training complete.")
        print(f"Areas under ROC Curves for each epoch: \n{AUCs}")
        return epoch_losses

def run(epochs, splits_path, splits_name, goal="classification", load_model=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print(f"Running on {device}.")

    # Build dataset
    t0=time.time()
    train_loader, test_loader, train_files, test_files = load_data.get_data(dataset_path=dataset_path, batch_size=32, splits_path = splits_path, splits_name = splits_name, num_workers = 0, goal = goal)
    print(f"Dataset loaded in {time.time()-t0:.3f}s")

    # Build aggregating dataset for evaluation
    aggregator = aggregate.AggregatorDataset(test_files)
    
    # Build or load model
    model = models.ResNet50(pretrained=True, goal=goal)
    models_path = os.path.join('models', 'resnet50')
    if load_model:
        model.load_state_dict(torch.load(os.path.join(models_path, load_model)))
        print("Loaded model", load_model)

    model.to(device) # puts model on GPU / CPU
    modelManager = ModelManager(model, models_path, device, train_loader, test_loader, aggregator, goal)
    modelManager.train(epochs, verbose=False)

if __name__ == "__main__": 
    epochs = 5
    # goal = "regression"
    goal = "classification"
    splits_path = "splits"
    dataset_path="dataset_cut_1"
    splits_name = models.name_model(epochs)
    run(epochs, splits_path, splits_name, goal=goal)