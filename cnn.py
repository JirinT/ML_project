import torch.nn as nn

from torch import flatten

class CNN(nn.Module):
    def __init__(self, config):
        """
        define the CNN architecture
        """
        super(CNN, self).__init__()

        num_channels = 1 if config["preprocessor"]["setting"]["rgb2gray"] == True or config["preprocessor"]["setting"]["rgb2lab"] == True else 3
        output_size = config["cnn"]["model"]["num_classes"]

        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=32*16*16, out_features=256)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(256, output_size)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        """
        forward pass of the CNN
        Args:
            x (torch.Tensor): The input data.

        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = flatten(x, 1)
        
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        output = self.logSoftmax(x)
        
        return output
