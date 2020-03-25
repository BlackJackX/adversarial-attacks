from __future__ import print_function   # 从future版本导入print函数功能
import argparse                         # 加载处理命令行参数的库
import torch                            # 引入相关的包
import torch.nn as nn                   # 指定torch.nn别名nn
import torch.nn.functional as F         # 引用神经网络常用函数包，不具有可学习的参数
import torch.optim as optim
import torch.nn.init as init
from torchvision import datasets, transforms  # 加载pytorch官方提供的dataset
import pickle
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1)  # 1表示输入通道，20表示输出通道，5表示conv核大小，1表示conv步长
        init.kaiming_normal_(self.conv1.weight)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        init.kaiming_normal_(self.conv2.weight)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        init.kaiming_normal_(self.conv3.weight)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        init.kaiming_normal_(self.conv4.weight)
        self.conv5 = nn.Conv2d(128, 256, 3, 1)
        init.kaiming_normal_(self.conv5.weight)
        self.conv6 = nn.Conv2d(256, 256, 3, 1)
        init.kaiming_normal_(self.conv6.weight)
        self.fc1 = nn.Linear(3 * 3 * 256, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 10)


    def forward(self, x):
        x = (x - torch.mean(x)) / torch.var(x)
        x = F.pad(x, (1, 1, 1, 1))
        x = F.relu(self.conv1(x))
        x = F.pad(x, (1, 1, 1, 1))
        x = F.relu(self.conv2(x))
        x = F.layer_norm(x, x.shape)
        x = F.max_pool2d(x, 2, 2)
        x = F.dropout2d(x, 0.3)

        x = F.pad(x, (1, 1, 1, 1))
        x = F.relu(self.conv3(x))
        x = F.pad(x, (1, 1, 1, 1))
        x = F.relu(self.conv4(x))
        x = F.layer_norm(x, x.shape)
        x = F.max_pool2d(x, 2, 2)
        x = F.dropout2d(x, 0.3)

        x = F.pad(x, (1, 1, 1, 1))
        x = F.relu(self.conv5(x))
        x = F.pad(x, (1, 1, 1, 1))
        x = F.relu(self.conv6(x))
        x = F.layer_norm(x, x.shape)
        x = F.max_pool2d(x, 2, 2)
        x = F.dropout2d(x, 0.3)

        x = x.view(-1, 3 * 3 * 256)
        x = self.fc1(x)
        x = F.dropout(x, 0.3)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    avg_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        avg_loss += loss
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} AvgLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                        100 * avg_loss/args.log_interval))
            avg_loss = 0.0


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def select_img(args, model, device, test_loader):
    model.eval()
    num = 0
    with torch.no_grad():
        for data, target in test_loader:
            if num >= 1000:
                break
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            for i, b in enumerate(pred.eq(target.view_as(pred))):
                if num >= 1000:
                    break
                if b:
                    num += 1
                    with open("./success_imgs/img"+str(num)+".pkl", "wb") as f:
                        pickle.dump(data[i], f)
                    print("Select %d images" %num)

    print("Select images success")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--model-url', type=str, default="", metavar='S',
                        help='Do you use the trained model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./fashionmnist_data/', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./fashionmnist_data/', train=False, transform=transforms.ToTensor()),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if not args.model_url:
        model = Net().to(device)

        #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        optimizer = optim.Adam(model.parameters())
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12, 24], gamma=0.1)

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader)

        if (args.save_model):
            torch.save(model.state_dict(), "./fashion_mnist_cnn.pt")

    else:
        model = Net().to(device)
        model.load_state_dict(torch.load(args.model_url))
        for epoch in range(1, args.epochs + 1):
            select_img(args, model, device, test_loader)

# 当.py文件直接运行时，该语句及以下的代码被执行，当.py被调用时，该语句及以下的代码不被执行
if __name__ == '__main__':
    main()