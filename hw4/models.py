import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

zvect = 100
feat=32
channels=3

class NetG(nn.Module):
    def __init__(self):
        super(NetG,self).__init__()
        # Adaptead from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        self.sequence = nn.Sequential(
            nn.ConvTranspose2d(zvect, feat * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feat * 4, feat * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( feat * 4, feat * 2 , 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( feat*2, feat, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat),
            nn.ReLU(True),
            nn.ConvTranspose2d( feat, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.sequence(input)

class NetD(nn.Module):
    def __init__(self):
        super(NetD, self).__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(channels, feat, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(feat, feat * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(feat * 2, feat * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(feat * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.sequence(input)
    

if __name__ == '__main__':
    trainset = datasets.CIFAR10(root='./hw4data', train=True, transform=None, download=True)
    testset = datasets.CIFAR10(root='./hw4data', train=False, transform=None, download=True)
    
    siz = len(testset)
    vals = {testset[x][1] for x in range(siz)}
    print(vals)

    gen = NetG()
    out = gen(torch.rand(5,100,1,1))
    print(out.shape)

    dis = NetD()
    out2 = dis(torch.rand(5,3,32,32)).squeeze()
    print(out2.shape)
    #print(dis)