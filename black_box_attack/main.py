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

eps = 2 * 8 / 225.
steps = 40
norm = float('inf')
step_alpha = 0.01
loss = nn.CrossEntropyLoss()

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


def targeted_attack(img, label):
    label = label.long()
    x, y = Variable(img, requires_grad=True), Variable(label)
    for step in range(steps):
        zero_gradients(x)
        out = white_model(torch.unsqueeze(x, dim=0))
        _loss = loss(out, torch.unsqueeze(y, dim=0))
        _loss.backward()
        normed_grad = step_alpha * torch.sign(x.grad.data)
        step_adv = x.data - normed_grad
        adv = step_adv - img
        adv = torch.clamp(adv, -eps, eps)
        result = img + adv
        result = torch.clamp(result, 0.0, 1.0)
        x.data = result
    return result, adv



def create_adv_img(img, label, times=2):
    while times >= 0:
        print("rested %d times" %times)
        adv_img, noise = targeted_attack(img, label)
        adv_img /= 255.
        adv_label = black_model(torch.unsqueeze(adv_img, dim=0)).max(1)[1]
        if int(adv_label[0]) == int(label):
            print(torch.dist(img, adv_img))
            return True, (adv_img, noise)
        times -= 1
    return False, None


reverse_trans = lambda x: np.asarray(transforms.ToPILImage()(x))

def draw_result(img, noise, adv_img, num, img_label, adv_label):
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    orig_class, attack_class = img_label, adv_label

    ax[0].imshow(reverse_trans(img[0].cpu()))
    ax[0].set_title('Original image: {}'.format(str(orig_class)))
    ax[1].imshow(reverse_trans(noise[0].cpu()))
    ax[1].set_title('Attacking noise')
    ax[2].imshow(reverse_trans(adv_img[0].cpu()))
    ax[2].set_title('Adversarial example: {}'.format(str(attack_class)))

    for i in range(3):
        ax[i].set_axis_off()
    plt.tight_layout()
    plt.savefig('./created_imgs/'+str(num))

if __name__ == "__main__":
    correct = 0
    for i in range(len(imgs)):
        print("No. %d" %i)
        try:
            adv_label = (labels[i] + 1) % 10
            flag, result = create_adv_img(imgs[i], adv_label)
            if result:
                adv_img, noise = result
            if flag:
                correct += 1
                draw_result(imgs[i], noise, adv_img, i, labels[i], adv_label)
        except Exception as e:
            print(str(e))

    print(correct / 1000)

