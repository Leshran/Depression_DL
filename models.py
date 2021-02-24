import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.models
import torchvision.models
from torchsummary import summary
import copy
import datetime
import os

def name_model(epochs):
    now = datetime.datetime.now()
    now_str = now.strftime("%Y_%m_%d_%H_%M")
    return f"{str(epochs)}_{now_str}".replace(' ', '_')

def save_model(model, models_path, epoch):
    os.makedirs(models_path, exist_ok=True)
    name = name_model(epoch)
    path = os.path.join(models_path, name)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}.")

'''
First convolutional model
Somewhat hopeless
'''
NUM_CONV_1=10 # try 32
NUM_CONV_2=20 # try 64
NUM_FC=500 # try 1024
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(4 * 12 * 12, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

def cnn():
    model = Net()
    return model

class Identity(nn.Module):
    def __init__(self, goal):
        super().__init__()
        if goal == "classification":
            self.fc = nn.Sequential(
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 1),
                        nn.Sigmoid())
        else:
            self.fc = nn.Sequential(
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Dropout(p=0.3),
                        nn.Linear(256, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                        nn.ReLU())

    def forward(self, x):
        x = self.fc(x)
        return x

def resnet34(goal):
    # TODO: think of a way such that predictions start at random between 0 and 1
    model = torchvision.models.resnet34(pretrained=False)
    # for param in model.parameters():
        # param.requires_grad = False
    new_fc = Identity(goal)
    model.fc = new_fc # Replace last layer with a custom one
    return model

class ResNet50(nn.Module):
    def __init__(self, pretrained = True, goal = "classification"):
        super(ResNet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained = pretrained)
        modules=list(self.resnet50.children())[:-1] # Drop last layer
        self.resnet50=nn.Sequential(*modules)
        if goal=="classification":
            self.fc = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(2048, 256),
                        nn.LeakyReLU(),
                        nn.Dropout(p=0.3),
                        nn.Linear(256, 256),
                        nn.LeakyReLU(),
                        nn.Dropout(p=0.3),
                        nn.Linear(256, 1),
                        nn.Sigmoid())
        else:
            self.fc = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(2048, 256),
                        nn.LeakyReLU(),
                        nn.Dropout(p=0.3),
                        nn.Linear(256, 256),
                        nn.LeakyReLU(),
                        nn.Dropout(p=0.3),
                        nn.Linear(256, 1),
                        nn.ReLU())
    
        # This lets us freeze the first parameters, but the trick of using different learning rates makes it unnecessary
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         if "resnet50" in name:
        #             param.requires_grad = False
                    # print("Freezing", name)
                # else:
                    # print("Not freezing", name)
    def forward(self, x):
        x = self.resnet50(x)
        x = self.fc(x)
        return x

##### Squeezenet #####
# Loading the Squeezenet model from TorchHub
class SqueezeNetClassifier(nn.Module):
    def __init__(self, pretrained = True, goal="classification"):
        super(SqueezeNetClassifier, self).__init__()
        self.squeezenet = torch.hub.load('pytorch/vision:v0.6.0', 'squeezenet1_1', pretrained)
        if goal == "classification":
            self.fc = nn.Sequential(
                        nn.Linear(1000, 64),
                        nn.LeakyReLU(),
                        nn.Dropout(p=0.3),
                        nn.Linear(64, 1),
                        nn.Sigmoid())
        else:
            self.fc = nn.Sequential(
                        nn.Linear(1000, 64),
                        nn.LeakyReLU(),
                        nn.Dropout(p=0.3),
                        nn.Linear(64, 1),
                        nn.ReLU())
        

    def forward(self, x):
        x = self.squeezenet(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    goal = "classification"
    model = ResNet50(pretrained = True)
    model.to(device)

    # for name, param in model.named_parameters():
        # print(name, param.data)
    summary(model, (3, 224,224))

    # model = resnet50(goal)
    # model.to(device)
    # summary(model, (3, 224, 224))