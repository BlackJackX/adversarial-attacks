import torch
from white_box_attack import  *
import pickle
import os
from torch import nn



if __name__ == "__main__":
    print(os.path.abspath('.'))
    with open("./attack_data/correct_1k.pkl", "rb") as f:
        data = pickle.load(f)
    model = torch.load('./Res18_accuracy_0.9264.pkl')
    #model.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
    x = torch.tensor(data[0]).cuda()
    print(model(x))