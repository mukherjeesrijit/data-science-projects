# The original UNet architecture consists of an input image of size 572x572. 
#Then, taking advantage of the beautiful powers of 2 encoded in the image size, 
#the authors created a beautiful U-shaped architecture with a segmentation mask output of size 388x388. 
#This is not the case most of the time, however. Often people want segmentation masks of the same size as the input. 
#Also, the powers of 2 and the multiples of 2 requirements caused by max-pooling should be a pain. 
#How to deal with images of odd dimensions becomes a natural question. 
#The UNet architecture is extended to a general architecture that can allow any input size and output the same size as the input. 
#Here are the details, I have added small perturbations to the original architecture. 
#Although there have been suggestions that change the input image's dimensions to the closest possible power of 2, 
#I think that leads to a large computational error. The following architecture has been compared with a few other options and found to be the fastest. 
#Original Architecture is given as follows: Group: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/, Arxiv: https://arxiv.org/abs/1505.04597

import torch
import torch.nn as nn
import torch.nn.functional as F

class Double_Convolution(nn.Module):
    def __init__(self, ic, oc):
        super(Double_Convolution, self).__init__()

        '''
        The double convolution blocks are taken with padding = 1. 
        This leads to the input and output of the convolutional layer being of the same size in terms of image size. 
        The channel dimensions of course change.
        '''

        self.double_convolution = nn.Sequential(
              nn.Conv2d(ic, oc, 3, 1, 1),
              nn.BatchNorm2d(oc),
              nn.ReLU(inplace = True),
              nn.Conv2d(oc, oc, 3, 1, 1),
              nn.BatchNorm2d(oc),
              nn.ReLU(inplace = True)
        )

    def forward(self, x):

        x = self.double_convolution(x)

        return x

class Down_Block(nn.Module):
    def __init__(self, ic, oc):
        super(Down_Block, self).__init__()

        '''
        The next change is in handling the odd size. 
        If the input to the max pool layer has an odd dimension, one zero padding layer is added to just that specific dimension. 
        This changes the odd dimension to just the next even dimension. Then max pool reduces the dimension by two as in the original architecture.
        '''

        self.downsample = nn.MaxPool2d(2)
        self.double_conv = Double_Convolution(ic, oc)

    def pad(self, x):
        _, _, h, w = x.shape

        # Calculate the amount of padding needed
        pad_h = (h % 2 != 0)
        pad_w = (w % 2 != 0)

        # Pad if necessary
        if pad_h or pad_w:
            padding = (0, int(pad_w), 0, int(pad_h))  # (left, right, top, bottom)
            x = F.pad(x, padding, mode='constant', value=0)

        return x

    def forward(self, x):
        x = self.pad(x)
        x = self.downsample(x)
        x = self.double_conv(x)

        return x

class Up_Block(nn.Module):
    def __init__(self, ic, oc):
        super(Up_Block, self).__init__()

        '''
        During upscaling, instead of upscaling directly by 2 by bilinear interpolation, the image is upscaled to the size "to be" concatenated feature by the skip                  connection using the interpolation function. Then a convolution with a 2x2 kernel is followed just like the original architecture. However, this reduces the 
        entire size by 1. Hence, padding is added exactly in the opposite direction of the padding in the down blocks, because this leads to symmetry of the architecture.
        Then the same double convolution of 3x3 kernel with padding = 1 is used, which retains the size of the input and output of the convolution layers.
        '''

        self.conv = nn.Conv2d(ic, oc, 2, 1)
        self.double_conv = Double_Convolution(ic, oc)

    def forward(self, x1, x2):

        _, _, h_in, w_in = x1.shape
        _, _, h_out, w_out = x2.shape

        x1 = F.interpolate(x1, size=(h_out, w_out), mode='bilinear', align_corners=False)

        padding = (1, 0, 1, 0)  # (left, right, top, bottom)
        x1 = F.pad(x1, padding, mode='constant', value=0)

        x1 = self.conv(x1)
        x = torch.cat((x1,x2), dim = 1)
        x = self.double_conv(x)

        return x

class UNet(nn.Module):
    def __init__(self, ic, oc):
        super(UNet, self).__init__()

        self.inconv = Double_Convolution(ic, 64)

        self.downconv1 = Down_Block(64,128)
        self.downconv2 = Down_Block(128,256)
        self.downconv3 = Down_Block(256,512)
        self.downconv4 = Down_Block(512,1024)

        self.upconv4 = Up_Block(1024,512)
        self.upconv3 = Up_Block(512, 256)
        self.upconv2 = Up_Block(256, 128)
        self.upconv1 = Up_Block(128, 64)

        self.outconv = nn.Conv2d(64, oc, 1, 1)

    def forward(self, x):

        print(x.shape)
        x1 = self.inconv(x)
        print(x1.shape)
        x2 = self.downconv1(x1)
        print(x2.shape)
        x3 = self.downconv2(x2)
        print(x3.shape)
        x4 = self.downconv3(x3)
        print(x4.shape)
        x5 = self.downconv4(x4)
        print(x5.shape)
        x6 = self.upconv4(x5, x4)
        print(x6.shape)
        x7 = self.upconv3(x6, x3)
        print(x7.shape)
        x8 = self.upconv2(x7, x2)
        print(x8.shape)
        x9 = self.upconv1(x8, x1)
        print(x9.shape)
        x10 = self.outconv(x9)
        print(x10.shape)

        return x10

b = 1 #batchsize
c = 1 #channels
h = 1013 #width
w = 1041 #height
image = torch.rand(b,c,h,w)
print(image.shape)
model = UNet(1,3)
model.eval()
output = model(image)
