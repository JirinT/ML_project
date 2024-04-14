import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor

class MultiHeadNetwork(nn.Module):
    def __init__(self, config, backbone_last_layer):
        super().__init__()

        # shared layers:
        self.shared_layers = backbone_last_layer
        # number of heads:
        self.num_heads = config["cnn"]["model"]["num_heads"]
        # define head layers:
        # the input_size is the output of last resnet layer, for resnet18 and resnet32 it is 512, for resnet 50 its 2048
        self.heads = nn.ModuleList([self.output_head_nn(input_size=512, output_size=config["cnn"]["model"]["num_classes"]) for _ in range(self.num_heads)])

    def output_head_nn(self, input_size, output_size):
        head = nn.Sequential(
            nn.Linear(input_size, input_size//2),
            nn.ReLU(),
            nn.Linear(input_size//2, output_size), # output_size must be set to 3 in config if we use this model for classification
        )
        return head
    
    def forward(self, x):
        # first pass through shared layers:
        x = self.shared_layers(x)

        x = x['AdaptiveAvgPool2d(output_size=(1, 1))'] # shape = (batch_size, 512, 1, 1)
        x = x.squeeze() # shape = (batch_size, 512)

        # pass through each head:
        x1 = self.heads[0](x)
        x2 = self.heads[1](x)
        x3 = self.heads[2](x)
        x4 = self.heads[3](x)

        return x1, x2, x3, x4