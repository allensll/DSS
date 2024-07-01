import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet_sm import resnet20, resnet32
from mobilenet_v2 import mobilenetv2
from torchvision.models import resnet18, resnet34, resnet50


class Shar(nn.Module):
    def __init__(self, arch='resnet20', n_party=2, num_classes=10):
        super(Shar, self).__init__()
        self.n_party = n_party
        self.num_classes = num_classes
        if arch in ['resnet20', 'resnet32']:
            self.nets = nn.ModuleList([eval(arch)(num_classes=num_classes) for i in range(n_party)])
        elif arch == 'mobilenetv2':
            self.nets = nn.ModuleList([mobilenetv2(num_classes=num_classes) for i in range(n_party)])
        elif arch in ['resnet18', 'resnet34', 'resnet50']:
            def builder():
                if num_classes == 200:
                    net = eval(arch)(pretrained=True)
                    net.avgpool = nn.AdaptiveAvgPool2d(1)
                    net.fc = nn.Linear(net.fc.in_features, 200)
                else:
                    net = eval(arch)(pretrained=False, num_classes=num_classes)
                return net
            self.nets = nn.ModuleList([builder() for i in range(n_party)])
        else:
            self.nets = nn.ModuleList([resnet20(num_classes=num_classes) for i in range(n_party)])

    def forward(self, x):
        y_sub = list()
        for i in range(self.n_party):
            y_sub.append(self.nets[i](x))
        y = sum(y_sub)
        return y_sub, y
