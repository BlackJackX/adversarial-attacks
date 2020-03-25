import os
import pickle
import torch
from my_model import Net
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

reverse_trans = lambda x: np.asarray(T.ToPILImage()(x))

loss = nn.CrossEntropyLoss()
net = Net()
net.load_state_dict(torch.load('./fashion_mnist_cnn.pt'))
net.cuda()

def load_success_img(num):
    with open(os.path.join('./success_imgs', 'img'+str(num)+'.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data




def non_targeted_attack(img, steps, step_alpha, eps):
    img = img.cuda()
    label = torch.zeros(1, 1).cuda()

    x, y = Variable(img, requires_grad=True), Variable(label)
    for step in range(steps):
        zero_gradients(x)
        out = net(x)
        y.data = out.data.max(1)[1]
        _loss = loss(out, y)
        _loss.backward()
        normed_grad = step_alpha * torch.sign(x.grad.data)
        step_adv = x.data + normed_grad
        adv = step_adv - img
        adv = torch.clamp(adv, -eps, eps)
        result = img + adv
        result = torch.clamp(result, 0.0, 1.0)
        x.data = result
    return result.cpu(), adv.cpu()

def get_class(img):
    x = Variable(img).cuda()
    cls = net(x).data.max(1)[1].cpu().numpy()[0]
    return "Class" + str(cls)

def draw_result(img, noise, adv_img):
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    orig_class, attack_class = get_class(img), get_class(adv_img)
    ax[0].imshow(reverse_trans(img[0]))
    ax[0].set_title('Original image: {}'.format(orig_class.split(',')[0]))
    ax[1].imshow(noise[0].cpu().numpy().transpose(1, 2, 0))
    ax[1].set_title('Attacking noise')
    ax[2].imshow(reverse_trans(adv_img[0]))
    ax[2].set_title('Adversarial example: {}'.format(attack_class))
    for i in range(3):
        ax[i].set_axis_off()
    plt.tight_layout()
    plt.show()



def main():
    data = load_success_img(10)
    print(get_class(data))
    draw_result(data, data, data)



if __name__ == "__main__":
    main()