import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        conv_kernel = (3,3)
        conv_stride = (1,1)
        conv_padding = 1

        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        self.upsamp = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.leaky_relu = nn.LeakyReLU()

        self.conv64 = self._conv_module(2, 64, conv_kernel, conv_stride, conv_padding,self.leaky_relu)
        self.conv128 = self._conv_module(64, 128, conv_kernel, conv_stride, conv_padding,self.leaky_relu)

        self.upsample1 = self._upsample_module(128,64,conv_kernel,conv_stride,conv_padding, self.upsamp, self.leaky_relu)
        self.upsample2 = self._upsample_module(64,32,conv_kernel,conv_stride,conv_padding, self.upsamp, self.leaky_relu)

        self.end_layer = self._exit_layer(32,1,conv_kernel,conv_stride,conv_padding,self.upsamp,self.leaky_relu)




    def forward(self,prev_img, next_img):
        x = torch.cat((prev_img,next_img), dim=1)

        x = self.conv64(x)
        x = self.pool(x)
        x = self.conv128(x)
        x = self.pool(x)

        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.pool(x)

        x = self.end_layer(x)

        return x 

    def _conv_module(self, in_channels, out_channels, kernel, stride, padding, leaky_relu):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel, stride, padding), leaky_relu,
            nn.Conv2d(in_channels, in_channels, kernel, stride, padding), leaky_relu,
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding), leaky_relu,
        )
    def _upsample_module(self, in_channels, out_channels, kernel, stride, padding, upsample, leaky_relu):
        return nn.Sequential(
            upsample, nn.Conv2d(in_channels,out_channels,kernel,stride,padding), leaky_relu
        )
    def _exit_layer(self,in_channels, out_channels,kernel,stride,padding,upsample,leaky_relu):
        return nn.Sequential(
            nn.Conv2d(in_channels,in_channels, kernel,stride,padding),leaky_relu,
            nn.Conv2d(in_channels,in_channels, kernel,stride,padding),leaky_relu,
            nn.Conv2d(in_channels,out_channels, kernel,stride,padding),leaky_relu,
            upsample,
            nn.Conv2d(out_channels,out_channels, kernel,stride,padding)
        )
