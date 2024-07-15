import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np
import torch.nn.functional as F

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()

        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Max pooling and average pooling along the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x = torch.cat([avg_out, max_out], dim=1)  # Concatenate along the channel dimension
        x = self.conv(x)  # 3D convolution
        att = self.sigmoid(x)  # Sigmoid activation to get attention map

        return att


class conv3D_block(nn.Module):

    def __init__(self, in_ch, out_ch):

        super(conv3D_block, self).__init__()

        self.conv3D = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.3),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.4)
        )

    def forward(self, x):
        x = self.conv3D(x)
        return x


class up_conv3D_block(nn.Module):

    def __init__(self, in_ch, out_ch, scale_tuple):

        super(up_conv3D_block, self).__init__()

        self.up_conv3D = nn.Sequential(
            nn.Upsample(scale_factor=scale_tuple, mode='trilinear'),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True), # increasing the depth by adding one below
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1), # no change in dimensions of 3D volume
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.4)
        )

    def forward(self, x):
        x = self.up_conv3D(x)
        return x

class Att_UNet(nn.Module):
    def __init__(self, in_ch_SA=3, out_ch_SA=4):
        super(Att_UNet, self).__init__()

        filters_3D = [16, 32, 64, 128, 256, 256]

        self.Conv3D_1 = conv3D_block(in_ch_SA, filters_3D[0])
        self.Conv3D_2 = conv3D_block(filters_3D[0], filters_3D[1])
        self.Conv3D_3 = conv3D_block(filters_3D[1], filters_3D[2])
        self.Conv3D_4 = conv3D_block(filters_3D[2], filters_3D[3])
        self.Conv3D_5 = conv3D_block(filters_3D[3], filters_3D[4])
        self.Conv3D_6 = conv3D_block(filters_3D[4], filters_3D[5])

        self.MaxPool3D_1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.MaxPool3D_2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.MaxPool3D_3 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.MaxPool3D_4 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.MaxPool3D_5 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

        self.up_Conv3D_1 = up_conv3D_block(filters_3D[5], filters_3D[4], (1,2,2))
        self.up_Conv3D_2 = up_conv3D_block(filters_3D[4] * 2, filters_3D[3], (1,2,2))
        self.up_Conv3D_3 = up_conv3D_block(filters_3D[3] * 2, filters_3D[2], (2,2,2))
        self.up_Conv3D_4 = up_conv3D_block(filters_3D[2] * 2, filters_3D[1], (2,2,2))
        self.up_Conv3D_5 = up_conv3D_block(filters_3D[1] * 2, filters_3D[0], (1,2,2))

        self.Conv3D_final = nn.Conv3d(filters_3D[0] * 2, out_ch_SA, kernel_size=1, stride=1, padding=0)

        self.spatial_attention = SpatialAttention3D(kernel_size=7)

    def forward(self, e_SA):
        # SA network's encoder
        e_SA_1 = self.Conv3D_1(e_SA)
        e_SA = self.MaxPool3D_1(e_SA_1)
        e_SA_2 = self.Conv3D_2(e_SA)
        e_SA = self.MaxPool3D_2(e_SA_2)
        e_SA_3 = self.Conv3D_3(e_SA)
        e_SA = self.MaxPool3D_3(e_SA_3)
        e_SA_4 = self.Conv3D_4(e_SA)
        e_SA = self.MaxPool3D_4(e_SA_4)
        e_SA_5 = self.Conv3D_5(e_SA)
        e_SA = self.MaxPool3D_5(e_SA_5)
        e_SA_6 = self.Conv3D_6(e_SA)

        del(e_SA)

        # 3D SPATIAL ATTENTION
        att = self.spatial_attention(e_SA_6)
        e_SA_6 = e_SA_6 * att

        # SA network's decoder
        d_SA = self.up_Conv3D_1(e_SA_6)
        d_SA = torch.cat([e_SA_5, d_SA], dim=1)
        d_SA = self.up_Conv3D_2(d_SA)
        d_SA = torch.cat([e_SA_4, d_SA], dim=1)
        d_SA = self.up_Conv3D_3(d_SA)
        d_SA = torch.cat([e_SA_3, d_SA], dim=1)
        d_SA = self.up_Conv3D_4(d_SA)
        d_SA = torch.cat([e_SA_2, d_SA], dim=1)
        d_SA = self.up_Conv3D_5(d_SA)
        d_SA = torch.cat([e_SA_1, d_SA], dim=1)
        d_SA = self.Conv3D_final(d_SA)

        del(e_SA_1, e_SA_2, e_SA_3, e_SA_4, e_SA_5)

        d_SA = F.softmax(d_SA, dim=1)

        return d_SA
    

def main():

    model = Att_UNet()

    #print("****** Model Summary ******")
    #summary(model, input_size = [(1, 1, 16, 256, 256)], batch_size = -1)

    #x = torch.rand(1, 1, 16, 256, 256)
    x = torch.rand(1, 3, 128, 128, 128)
    
    pred = model(x)
    print(pred.shape) # --> (1, 4, 128, 128, 128)


if __name__ == '__main__':

    main()
