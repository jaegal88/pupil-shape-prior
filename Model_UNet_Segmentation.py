import torch
import torch.nn as nn
import parameters

first_8 = 8
first_16 = 16
first_32 = 32
Size_X = parameters.Size_X
Size_Y = parameters.Size_Y

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet4f32ch_sigmoid(nn.Module):

    def __init__(self):
        super().__init__()
        self.first = 32

        self.dconv_down1 = double_conv(1, self.first)
        self.dconv_down2 = double_conv(self.first, self.first*2)
        self.dconv_down3 = double_conv(self.first*2, self.first*4)
        self.dconv_down4 = double_conv(self.first*4, self.first*8)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.upsample3 = nn.ConvTranspose2d(in_channels=self.first*8, out_channels=self.first*4, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(in_channels=self.first*4, out_channels=self.first*2, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(in_channels=self.first*2, out_channels=self.first, kernel_size=2, stride=2)

        self.dconv_up3 = double_conv(self.first*4 + self.first*4, self.first*4)
        self.dconv_up2 = double_conv(self.first*2 + self.first*2, self.first*2)
        self.dconv_up1 = double_conv(self.first + self.first, self.first)

        self.conv_last = nn.Conv2d(in_channels=self.first, out_channels=1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        d1 = self.maxpool1(conv1)

        conv2 = self.dconv_down2(d1)
        d2 = self.maxpool2(conv2)

        conv3 = self.dconv_down3(d2)
        d3 = self.maxpool3(conv3)

        conv4 = self.dconv_down4(d3)

        u3 = self.upsample3(conv4)
        cat3 = torch.cat([u3, conv3], dim=1)

        conv_3 = self.dconv_up3(cat3)
        u2 = self.upsample2(conv_3)
        cat2 = torch.cat([u2, conv2], dim=1)

        conv_2 = self.dconv_up2(cat2)
        u1 = self.upsample1(conv_2)
        cat1 = torch.cat([u1, conv1], dim=1)

        conv_1 = self.dconv_up1(cat1)
        ch_2 = self.conv_last(conv_1)
        out = self.sigmoid(ch_2)

        return out
