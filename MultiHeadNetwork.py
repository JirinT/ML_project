import torch.nn as nn
from torch import flatten

class MultiHeadNetwork(nn.Module):
    def __init__(self, config, shared_backbone):
        super().__init__()

        # shared layers:
        self.shared_layers = shared_backbone

        # number of heads:
        self.num_heads = config["cnn"]["num_heads"]

        # define head layers:
        self.heads = nn.ModuleList([self.output_head_nn(self.shared_layers.fc.in_features, config["cnn"]["model"]["num_classes"]) for _ in range(self.num_heads)])


    def output_head_nn(self, input_size, output_size):
        head = nn.Sequential(
            nn.Linear(input_size, input_size//2),
            nn.ReLU(),
            nn.Linear(input_size//2, output_size),
            nn.LogSoftmax(dim=1)
        )

        return head
    
    def forward(self, x):
        # first pass through shared layers:
        x = self.shared_layers(x)

        # pass through each head:
        x1 = self.heads[0](x)
        x2 = self.heads[1](x)
        x3 = self.heads[2](x)
        x4 = self.heads[3](x)

        return x1, x2, x3, x4