import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

class conv2D_block(nn.Module):

    def __init__(self, in_ch, out_ch):

        super(conv2D_block, self).__init__()

        self.conv2D = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3)
        )

    def forward(self, x):
        x = self.conv2D(x)
        return x

class up_conv2D_block(nn.Module):

    def __init__(self, in_ch, out_ch, scale_tuple):

        super(up_conv2D_block, self).__init__()

        self.up_conv2D = nn.Sequential(
            nn.Upsample(scale_factor=scale_tuple, mode='bilinear'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True), # increasing the depth by adding one below
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3)
        )

    def forward(self, x):
        x = self.up_conv2D(x)
        return x

class UNet_mod(nn.Module):

    def __init__(self, in_ch_SA=3, out_ch_SA=4):
        super(UNet_mod, self).__init__()

        filters_2D = [16, 32, 64, 128]

        self.Conv2D_1 = conv2D_block(in_ch_SA, filters_2D[0])
        self.Conv2D_2 = conv2D_block(filters_2D[0], filters_2D[1])
        self.Conv2D_3 = conv2D_block(filters_2D[1], filters_2D[2])
        self.Conv2D_4 = conv2D_block(filters_2D[2], filters_2D[3])

        self.MaxPool2D_1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.MaxPool2D_2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.MaxPool2D_3 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.up_Conv2D_1 = up_conv2D_block(filters_2D[3], filters_2D[2], (2,2))
        self.up_Conv2D_2 = up_conv2D_block(filters_2D[2]+filters_2D[2], filters_2D[1], (2,2))
        self.up_Conv2D_3 = up_conv2D_block(filters_2D[1]+filters_2D[1], filters_2D[0], (2,2))

        self.Conv2D_final = nn.Conv2d(filters_2D[0]+filters_2D[0], out_ch_SA, kernel_size=1, stride=1, padding=0)

    def forward(self, e_SA):

        # SA network's encoder
        e_SA_1 = self.Conv2D_1(e_SA)
        #print("E1:", e_SA_1.shape)
        e_SA = self.MaxPool2D_1(e_SA_1)
        #print("E2:", e_SA.shape)
        e_SA_2 = self.Conv2D_2(e_SA)
        #print("E3:", e_SA_2.shape)
        e_SA = self.MaxPool2D_2(e_SA_2)
        #print("E4:", e_SA.shape)
        e_SA_3 = self.Conv2D_3(e_SA)
        #print("E5:", e_SA_3.shape)
        e_SA = self.MaxPool2D_3(e_SA_3)
        #print("E6:", e_SA.shape)
        e_SA_4 = self.Conv2D_4(e_SA)

        del(e_SA)

        # SA network's decoder
        d_SA = self.up_Conv2D_1(e_SA_4)
        #print("E11:", e_SA_6.shape)
        #print("D1:", d_SA.shape)
        d_SA = torch.cat([e_SA_3, d_SA], dim=1)
        #print("D2:", d_SA.shape)
        d_SA = self.up_Conv2D_2(d_SA)
        #print("D3:", d_SA.shape)
        d_SA = torch.cat([e_SA_2, d_SA], dim=1)
        #print("D4:", d_SA.shape)
        d_SA = self.up_Conv2D_3(d_SA)
        #print("D5:", d_SA.shape)
        d_SA = torch.cat([e_SA_1, d_SA], dim=1)

        d_SA = self.Conv2D_final(d_SA)
        #print("D11:", d_SA.shape)

        del(e_SA_1, e_SA_2, e_SA_3, e_SA_4)
        d_SA = F.softmax(d_SA, dim=1)

        return d_SA