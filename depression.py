import warnings
warnings.simplefilter("ignore", UserWarning) # Pytorch + multiprocessing means a lot of warnings
import load_data
import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.autograd import Variable
import os
import time
import datetime

class ModelManager():
    def __init__(self, model, models_path, device, train_loader, test_loader, goal):
        self.model = model
        self.models_path = models_path
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.goal = goal
        if goal == "classification":
            # self.criterion = nn.BCEWithLogitsLoss() # Classification - Binary Cross Entropy, converted to the [0, 1] range
            self.criterion = nn.BCELoss() # Classification - Binary Cross Entropy, converted to the [0, 1] range
        else:
            self.criterion = nn.MSELoss() # Regression - Mean Squared Error
    
    def train(self, epochs, verbose=False):
        self.model.train()
        optimizer = torch.optim.AdamW([
            {'params': self.model.resnet50.parameters(), 'lr': 0.0005}, # Low learning rate for the pretrained layers
            {'params': self.model.fc.parameters(), 'lr': 0.001}  # Higher learning rate for the final classifying layers
        ])
        
        epoch_losses = []
        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            epoch_start_time = time.time()
            batch_start_time = time.time()
            total_y = 0
            total_samples = 0
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
                batch_start_time = time.time()
                print(f"Epoch {epoch} - ({step*len(X)}/{len(self.train_loader.dataset)}) - Loss: {loss.item():.3f} - {batch_duration:.3f}s")
                if step % 10 == 0 and verbose:
                    print(f"Epoch {epoch} - ({step*len(X)}/{len(self.train_loader.dataset)}) - Last prediction: {prediction} vs {y}")
            epoch_time = time.time() - epoch_start_time
            epoch_losses.append(epoch_loss)
            print(f"Epoch {epoch} done. Average loss: {(epoch_loss/len(self.train_loader.dataset)):.3f} - {epoch_time:.4f}s")
            if verbose:
                print("Last prediction", prediction)
                print("Last y", y)
            models.save_model(self.model, self.models_path, epoch)
        models.save_model(self.model, self.models_path, epoch)

        return epoch_losses

    def test(self):
        self.model.eval()
        correct = 0
        eval_start_time = time.time()
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.float(), y.float() # Convert as they're stored as doubles
                X, y = Variable(X).to(self.device), Variable(y).to(self.device) # Bring to GPU
                out = self.model(X)     # input x and predict based on x
                prediction = out.round()
                # print(out, y)
                correct += prediction.eq(y).sum().item()
        taux_classif = 100. * correct / len(self.test_loader.dataset)
        print('Accuracy: {}/{} (tx {:.2f}%, err {:.2f}%)\n'.format(correct, len(self.test_loader.dataset), taux_classif, 100.-taux_classif))
        print(f"Evaluation done in {time.time() - eval_start_time}")

def run(epochs, splits_path, splits_name, goal="classification", load_model=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print(f"Running on {device}.")

    t0=time.time()
    train_loader, test_loader = load_data.get_data(batch_size=16, splits_path = splits_path, splits_name = splits_name, num_workers = 0, goal = goal)
    print(f"Dataset loaded in {time.time()-t0:.3f}s")
    
    # model = models.cnn()
    model = models.ResNet50(pretrained=True, goal=goal)
    models_path = os.path.join('models', 'resnet50')
    if load_model:
        model.load_state_dict(torch.load(os.path.join(models_path, load_model)))
        print("Loaded model", load_model)

    model.to(device) # puts model on GPU / CPU
    modelManager = ModelManager(model, models_path, device, train_loader, test_loader, goal)
    modelManager.train(epochs, verbose=False)
    modelManager.test()

if __name__ == "__main__": 
    epochs = 20
    # goal = "regression"
    goal = "classification"
    splits_path = "splits"
    splits_name = models.name_model(epochs)
    run(epochs, splits_path, splits_name, goal=goal)