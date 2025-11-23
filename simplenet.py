import torch, torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import cv2
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import ResNet50_Weights

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), 
    transforms.ToTensor()
])


train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               download=True,
                               transform=transform)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class Simplenet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        backbone = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Extract layers 2 and 3 of 
        return_nodes = {
            "layer2": "feat2",
            "layer3": "feat3",
        }
        self.backbone = create_feature_extractor(backbone, return_nodes=return_nodes)

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Sabemos que layer3 de ResNet50 tiene 1024 canales
        in_feats = 1024

        # Pool + MLP de dos capas
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_feats, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        out = self.backbone(x)   # dict con las keys que hemos definido
        feat2 = out["feat2"]
        feat3 = out["feat3"]
        return feat2, feat3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    simplenet = Simplenet().to(device)
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        output = simplenet.forward(data)
    pass    
