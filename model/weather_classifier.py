import torch
from torch import nn
from torchvision.models import resnet152

weather_model = resnet152()
weather_model.fc = nn.Linear(2048, 11)
weather_model.load_state_dict(torch.load("weather.pt"))
