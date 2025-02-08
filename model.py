import torch
import torch.nn as nn
from torchvision import models

class DeepFakeDetector(nn.Module):
    def __init__(self):
        super(DeepFakeDetector, self).__init__()
        self.model = models.mobilenet_v3_small(pretrained=True)
        self.model.classifier[3] = nn.Linear(1024, 1)  # Binary classification

    def forward(self, x):
        return self.model(x)

def load_model():
    model = DeepFakeDetector()
    model.load_state_dict(torch.load("deepfake_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model
