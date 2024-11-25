import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

zvect = 100
feat=64
feat2 = 64
# Used to be 32
channels=3

class NetG(nn.Module):
    def __init__(self):
        super(NetG,self).__init__()
        self.embedding = nn.Embedding(10,100)
        
        # Adaptead from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        """ self.sequence = nn.Sequential(
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
        ) """

        self.sequence = nn.Sequential(
            nn.ConvTranspose2d(zvect, feat * 8, 4,1,0, bias=False),
            nn.BatchNorm2d(feat * 8, momentum =.9),
            nn.ReLU(True),
            nn.ConvTranspose2d(feat * 8, feat * 4, 4,2,1, bias=False),
            nn.BatchNorm2d(feat * 4, momentum=.9),
            nn.ReLU(True),
            nn.ConvTranspose2d( feat * 4, feat * 2, 4,2,1,bias=False),
            nn.BatchNorm2d(feat * 2, momentum=.9),
            nn.ReLU(True),
            nn.ConvTranspose2d( feat*2, feat, 4,2,1, bias=False),
            nn.BatchNorm2d(feat, momentum=.9),
            nn.ReLU(True),
            nn.ConvTranspose2d( feat, channels, 4,2,1, bias=False),
            nn.Tanh(),
        )

    def forward(self, inputs, label):
        bs = inputs.size(0)
        if len(label.size()) == 1:
            label = label.unsqueeze(-1)
        # print(f'{label.shape = }')
        # embed = nn.Embedding(10, 100)
        labels = self.embedding(label).view(bs, 100, 1,1)
        #print(f'{inputs.shape = }')
        #print(f'{labels.shape = }')
        #inputs = torch.concat([inputs, labels.float()], dim=1)
        #inputs = nn.ReLU()(inputs)
        return self.sequence(inputs)

class NetD(nn.Module):
    def __init__(self):
        super(NetD, self).__init__()
        self.embedding = nn.Embedding(10, feat2*8)
        self.sequence = nn.Sequential(
            nn.Conv2d(channels, feat2, 4,2,1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(feat2, feat2 * 2, 4,2,1, bias=False),
            nn.BatchNorm2d(feat2 * 2, momentum=.9),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(feat2 * 2, feat2 * 4, 4,2,1, bias=False),
            nn.BatchNorm2d(feat2 * 4, momentum =.9),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(feat2 * 4, feat2 * 8, 4,2,1, bias=False),
            nn.BatchNorm2d(feat2 * 8, momentum=.9),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Conv2d(feat2 * 8, 1, 2, 1, 0, bias=False),
            #nn.Sigmoid()
        )
        self.last_conv2d = nn.Conv2d(feat2 * 8, 1, 4,1,0, bias=False)
        self.flatten = nn.Flatten()
        #self.dropout = nn.Dropout(.30)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, label):
        bs = inputs.size(0)
        if len(label.size()) == 1:
            label = label.unsqueeze(-1)
        label = nn.ReLU()(self.embedding(label))
        # print(f'{label.shape = }')
        label = label.unsqueeze(1)
        # print(f'{label.shape = }')
        label = label.tile((1,4,4,1))
        #print(f'{label.shape = }')

        output =  self.sequence(inputs)
        #print(f'{output.shape = }')
        label = torch.transpose(label,3, 1)
        #print(f'{label.shape = }')
        
        #output = torch.concat([output, label], dim=1)
        # print(f'{output.shape = }')
        output = self.last_conv2d(output)
        output = self.flatten(output)
        #output = self.dropout(output)
        output = self.sigmoid(output)
        return output
        
    
    

if __name__ == '__main__':
    trainset = datasets.CIFAR10(root='./hw4data', train=True, transform=None, download=True)
    testset = datasets.CIFAR10(root='./hw4data', train=False, transform=None, download=True)
    
    siz = len(testset)
    vals = {testset[x][1] for x in range(siz)}
    print(vals)

    gen = NetG()
    out = gen(torch.randn(5,100,1,1), torch.randint(0,10,(5,1)))
    print(out.shape)

    dis = NetD()
    out2 = dis(torch.rand(5,3,64,64), torch.randint(0,10,(5,))).squeeze()
    print(out2.shape)
    print(out2)
    #print(dis)
    