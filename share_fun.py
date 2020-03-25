import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.models.inception import inception_v3

from PIL import Image
#from scipy.misc import imsave
import imageio

import matplotlib.pyplot as plt
import os
import numpy as np


def draw_img(img, label):
    plt.figure("image label: %s" %label)
    plt.title("image label: %s" %label)
    plt.imshow(img)
    plt.show()




if __name__ == "__main__":
    import mnist

    dataset_loader = mnist.MNIST('./data', return_type='numpy')

    dataset_loader.load_training()
    dataset_loader.load_testing()

    draw_img(dataset_loader.train_images[0].reshape(28,28), dataset_loader.train_labels[0])
