import torch
import torch.nn as nn
import torch.nn.functional as F

def double_conv(in_channels, out_channels):
    # returns a block compsed of two Convolution layers with ReLU activation function
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU()
    )   

class DownSampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = ...
        self.maxpool = ...

    def forward(self, x):
        x_skip = ...
        out = ... 

        return out , x_skip

class UpSampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = ...
        self.upsample = ... # use nn.Upsample

    def forward(self, x, x_skip):
        x = self.upsample(x)
        x = torch.cat([x, x_skip], dim=1) # concatenates x and x_skip
        x = self.conv_block(x)
        
        return x
    

class UNet(nn.Module):

    def __init__(self):
        super().__init__()
                
        self.downsample_block_1 = ...
        self.downsample_block_2 = ...
        self.downsample_block_3 = ...
        self.middle_conv_block = double_conv(128, 256)        

            
        self.upsample_block_3 = ...
        self.upsample_block_2 = ...
        self.upsample_block_1 = ...
        
        self.last_conv = nn.Conv2d(32, 3, 1)
        
        
    def forward(self, x):
        x, x_skip1 = ...
        x, x_skip2 = ...
        x, x_skip3 = ... 
        
        x = self.middle_conv_block(x)
        
        x = #use upsampleblock_3 and x_skip3
        x = #use upsampleblock_2 and x_skip2
        x = #use upsampleblock_1 and x_skip1       
        
        out = self.last_conv(x)
        
        return out
    
    def get_features(self, x):
        x, _ = self.downsample_block_1(x)
        x, _ = self.downsample_block_2(x)
        x, _ = self.downsample_block_3(x)
        return x

        
if __name__=='__main__':
    x = torch.rand(16,1,224,224)
    net = UNet()
    y = net(x)
    assert y.shape == (16,3,224,224)
    print('Shapes OK')
