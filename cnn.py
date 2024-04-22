import torch.nn as nn

class CNN2(nn.Module):
    def __init__(self, config):
        """
        define the CNN architecture
        """
        super(CNN2, self).__init__()

        self.config = config

        num_channels = 1 if config["preprocessor"]["setting"]["rgb2gray"] == True or config["preprocessor"]["setting"]["rgb2lab"] == True else 3
        input_width = config["preprocessor"]["resize"]["width"]
        input_height = config["preprocessor"]["resize"]["height"]
        output_size = config["cnn"]["model"]["num_classes"]
        droupout_rate = config["cnn"]["model"]["dropout_rate"]

        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        if config["cnn"]["training"]["normalization"]["use_batch_normalization"]:
            self.bn1 = nn.BatchNorm2d(16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        if config["cnn"]["training"]["normalization"]["use_batch_normalization"]:
            self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        width = input_width
        height = input_height
        for layer in [self.conv1, self.pool1, self.conv2, self.pool2]:
            if isinstance(layer.kernel_size, int):
                width = (width - layer.kernel_size + 2*layer.padding) // layer.stride + 1
                height = (height - layer.kernel_size + 2*layer.padding) // layer.stride + 1
            else:
                width = (width - layer.kernel_size[0] + 2*layer.padding[0]) // layer.stride[0] + 1
                height = (height - layer.kernel_size[1] + 2*layer.padding[1]) // layer.stride[1] + 1

        self.flatten = nn.Flatten(start_dim=1)

        self.fc1 = nn.Linear(in_features=32*width*height, out_features=256)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=droupout_rate)

        self.fc2 = nn.Linear(256, output_size)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        forward pass of the CNN
        Args:
            x (torch.Tensor): The input data.
        """
        x = self.conv1(x)
        if self.config["cnn"]["training"]["normalization"]["use_batch_normalization"]:
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        if self.config["cnn"]["training"]["normalization"]["use_batch_normalization"]:
            x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        output = self.logSoftmax(x)
        
        return output
    

class CNN4(nn.Module):
    def __init__(self, config):
        """
        define the CNN architecture
        """
        super(CNN4, self).__init__()

        self.config = config

        num_channels = 1 if config["preprocessor"]["setting"]["rgb2gray"] == True or config["preprocessor"]["setting"]["rgb2lab"] == True else 3
        input_width = config["preprocessor"]["resize"]["width"]
        input_height = config["preprocessor"]["resize"]["height"]
        output_size = config["cnn"]["model"]["num_classes"]
        droupout_rate = config["cnn"]["model"]["dropout_rate"]

        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        if config["cnn"]["training"]["normalization"]["use_batch_normalization"]:
            self.bn1 = nn.BatchNorm2d(16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        if config["cnn"]["training"]["normalization"]["use_batch_normalization"]:
            self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        if config["cnn"]["training"]["normalization"]["use_batch_normalization"]:
            self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        if config["cnn"]["training"]["normalization"]["use_batch_normalization"]:
            self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        width = input_width
        height = input_height
        for layer in [self.conv1, self.pool1, self.conv2, self.pool2, self.conv3, self.pool3, self.conv4, self.pool4]:
            if isinstance(layer.kernel_size, int):
                width = (width - layer.kernel_size + 2*layer.padding) // layer.stride + 1
                height = (height - layer.kernel_size + 2*layer.padding) // layer.stride + 1
            else:
                width = (width - layer.kernel_size[0] + 2*layer.padding[0]) // layer.stride[0] + 1
                height = (height - layer.kernel_size[1] + 2*layer.padding[1]) // layer.stride[1] + 1

        self.flatten = nn.Flatten(start_dim=1)

        self.fc1 = nn.Linear(in_features=128*width*height, out_features=256)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=droupout_rate)

        self.fc2 = nn.Linear(256, output_size)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        forward pass of the CNN
        Args:
            x (torch.Tensor): The input data.
        """
        x = self.conv1(x)
        if self.config["cnn"]["training"]["normalization"]["use_batch_normalization"]:
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        if self.config["cnn"]["training"]["normalization"]["use_batch_normalization"]:
            x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        if self.config["cnn"]["training"]["normalization"]["use_batch_normalization"]:
            x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        if self.config["cnn"]["training"]["normalization"]["use_batch_normalization"]:
            x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        output = self.logSoftmax(x)
        
        return output