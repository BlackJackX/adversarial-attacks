import torch
from white_box_attack import  *
import pickle
import os
from torch import nn
from torchvision import transforms
from my_model import Net
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import matplotlib.pyplot as plt
import os
import numpy as np


"""------------ load models and data ---------------------"""
device = 'cuda: 0'

with open("./attack_data/correct_1k.pkl", "rb") as f:
    data = pickle.load(f)

black_model = torch.load('./Res18_accuracy_0.9264.pkl').to(device)
black_model.conv5_x = nn.AvgPool2d(7, 7, 0)

white_model = Net().to(device)
white_model.load_state_dict(torch.load('./fashion_mnist_cnn.pt'))

imgs = torch.tensor(data[0]).to(device)

labels = torch.tensor(data[1]).to(device).squeeze(dim=1)
labels = labels.max(1)[1]
"""-------------- white box attack ---------------------"""