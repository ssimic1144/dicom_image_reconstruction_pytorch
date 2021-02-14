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
        self.relu = nn.ReLU()

        self.conv64 = self._conv_module(2, 64, conv_kernel, conv_stride, conv_padding,self.relu)
        self.conv128 = self._conv_module(64, 128, conv_kernel, conv_stride, conv_padding,self.relu)
        self.conv256 = self._conv_module(128, 256, conv_kernel, conv_stride, conv_padding,self.relu)
        self.conv256x256 = self._conv_module(256, 256, conv_kernel, conv_stride, conv_padding,self.relu)

        self.upsample1 = self._upsample_module(256,256,conv_kernel,conv_stride,conv_padding, self.upsamp, self.relu)
        self.mid_conv1 = self._conv_module(256, 128, conv_kernel, conv_stride, conv_padding,self.relu)
        self.upsample2 = self._upsample_module(128,128,conv_kernel,conv_stride,conv_padding, self.upsamp, self.relu)
        self.mid_conv2 = self._conv_module(128, 64, conv_kernel, conv_stride, conv_padding,self.relu)
        self.upsample3 = self._upsample_module(64,32,conv_kernel,conv_stride,conv_padding, self.upsamp, self.relu)
        self.mid_conv3 = self._conv_module(32, 16, conv_kernel, conv_stride, conv_padding,self.relu)

        self.end_layer = self._exit_layer(16,1,conv_kernel,conv_stride,conv_padding,self.upsamp,self.relu)




    def forward(self,prev_img, next_img):
        x = torch.cat((prev_img,next_img), dim=1)

        x = self.conv64(x)
        x = self.pool(x)
        #print("first ",x.shape)
        x = self.conv128(x)
        x = self.pool(x)
        x = self.conv256(x)
        x = self.pool(x)
        #print("sec ",x.shape)
        x = self.conv256x256(x)
        x = self.pool(x)
        #print(" thrd",x.shape)

        x = self.upsample1(x)
        #print(" forth",x.shape)
        x = self.mid_conv1(x)
        #print(" fifth",x.shape)
        x = self.upsample2(x)
        #print(" sixth",x.shape)
        x = self.mid_conv2(x)
        x = self.upsample3(x)
        x = self.mid_conv3(x)
        
        
        x = self.end_layer(x)
        #print(x.shape)
        return x 

    def _conv_module(self, in_channels, out_channels, kernel, stride, padding, relu):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel, stride, padding), relu,
            nn.Conv2d(in_channels, in_channels, kernel, stride, padding), relu,
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding), relu,
        )
    def _upsample_module(self, in_channels, out_channels, kernel, stride, padding, upsample, relu):
        return nn.Sequential(
            upsample, nn.Conv2d(in_channels,out_channels,kernel,stride,padding), relu
        )
    def _exit_layer(self,in_channels, out_channels,kernel,stride,padding,upsample,relu):
        return nn.Sequential(
            nn.Conv2d(in_channels,in_channels, kernel,stride,padding),relu,
            nn.Conv2d(in_channels,in_channels, kernel,stride,padding),relu,
            nn.Conv2d(in_channels,out_channels, kernel,stride,padding),relu,
            upsample,
            nn.Conv2d(out_channels,out_channels, kernel,stride,padding),
        )
