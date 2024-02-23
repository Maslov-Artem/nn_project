import torch
from torch import nn
from torchvision.models import resnet152

bird_model = resnet152()
bird_model.fc = nn.Linear(2048, 200)
bird_model.load_state_dict(torch.load("birds.pt", map_location="cpu"))
