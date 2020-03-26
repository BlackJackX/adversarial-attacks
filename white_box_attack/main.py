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
from torchvision import datasets, transforms

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
net.eval()

eps = 2 * 8 / 225.
steps = 40
norm = float('inf')
step_alpha = 0.01

classes = {
    0:"T-shirt/top",
    1:"Trouser",
    2:"Pullover",
    3:"Dress",
    4:"Coat",
    5:"Sandal",
    6:"Shirt",
    7:"Sneaker",
    8:"Bag",
    9:"Ankle boot",
}

def load_success_img(num):
    with open(os.path.join('./success_imgs', 'img'+str(num)+'.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data



def non_targeted_attack(img, steps):
    img = img.cuda()
    label = torch.zeros(1, 1).cuda()

    x, y = Variable(img, requires_grad=True), Variable(label)
    l1 = get_class(img)
    l2 = -1
    step = 0
    while True:
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
        l2 = get_class(result)
        x.data = result

        step += 1
        if step > steps or l1 != l2:
            break

    return result.cpu(), adv.cpu()

def get_class(img):
    x = Variable(img).cuda()
    cls = net(x).data.max(1)[1].cpu().numpy()[0]
    return classes[cls]

def draw_result(img, noise, adv_img, num, img_label, adv_label):
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    orig_class, attack_class = img_label, adv_label

    ax[0].imshow(reverse_trans(img[0].cpu()))
    ax[0].set_title('Original image: {}'.format(orig_class.split(',')[0]))
    ax[1].imshow(reverse_trans(noise[0].cpu()))
    ax[1].set_title('Attacking noise')
    ax[2].imshow(reverse_trans(adv_img[0].cpu()))
    ax[2].set_title('Adversarial example: {}'.format(attack_class))

    for i in range(3):
        ax[i].set_axis_off()
    plt.tight_layout()
    plt.savefig('./created_imgs/'+str(num))


def draw_img(img, label):
    plt.figure("image label: %s" %label)
    plt.title("image label: %s" %label)
    plt.imshow(torch.squeeze(img.cpu()))
    plt.show()

def create_adv_img(num):
    data = load_success_img(num).cuda()
    img = torch.unsqueeze(data, dim=0)
    adv, noi = non_targeted_attack(img, 500)
    img_label = get_class(img)
    adv_label = get_class(adv)
    print(img_label, adv_label)
    if img_label == adv_label:
        print(img_label + ' failed')
        return False
    else:
        print(img_label + ' ' + adv_label + ' successed')
        draw_result(img, noi, adv, num, img_label, adv_label)
        return True


    #draw_result(img, noi[0], adv[0])


if __name__ == "__main__":
    correct = 0
    success_list = []
    for i in range(1, 1001):
        print("No. %d" %i)
        try:
            if create_adv_img(i):
                correct += 1
                success_list.append(i)
        except Exception as e:
            print(str(e))
    with open("success_list", "wb") as f:
        pickle.dump("success_list", f)
    print(correct / 1000)
