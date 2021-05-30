import torch
from torch import nn
from torchsummary import summary
from efficientnet_pytorch import EfficientNet

import config

class EfNetModel(nn.Module):
    def __init__(self, num_classes, dropout=0.2, pretrained_path=""):
        super().__init__()
        
        self.model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
        if not config.multi_channel:
            self.model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
        else:
            self.model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes, in_channels=6)

        if pretrained_path:
            self.load_state_dict(torch.load(pretrained_path))
        
        # set dropout
        self.model._dropout = nn.Dropout(dropout)
        
        for param in self.model.parameters():
            param.requires_grad = True

        # for param in self.model._fc.parameters():
        #     param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x