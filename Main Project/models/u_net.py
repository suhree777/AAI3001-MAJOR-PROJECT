import torch
import torch.nn as nn
import torch.nn.functional as F

# Defining a block for 2D convolution with optional batch normalization
class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, batchnorm=True):
        super(Conv2dBlock, self).__init__()

        # First 2D convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        # Second 2D convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        # Optional batch normalization layers
        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

    # Forward pass through the block
    def forward(self, x):
        x = self.conv1(x)
        
        # Applying batch normalization if enabled
        if self.batchnorm:
            x = self.bn1(x)
        
        # Applying ReLU activation function
        x = F.relu(x)

        x = self.conv2(x)
        
        # Applying batch normalization if enabled
        if self.batchnorm:
            x = self.bn2(x)
        
        # Applying ReLU activation function
        x = F.relu(x)

        return x

# Defining a U-Net architecture
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_filters=16, batchnorm=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_filters = n_filters

        # Contracting Path
        self.c1 = Conv2dBlock(n_channels, n_filters, batchnorm=batchnorm)
        self.c2 = Conv2dBlock(n_filters, n_filters*2, batchnorm=batchnorm)
        self.c3 = Conv2dBlock(n_filters*2, n_filters*4, batchnorm=batchnorm)
        self.c4 = Conv2dBlock(n_filters*4, n_filters*8, batchnorm=batchnorm)
        self.c5 = Conv2dBlock(n_filters*8, n_filters*16, batchnorm=batchnorm)

        # Expansive Path
        self.up6 = nn.ConvTranspose2d(n_filters*16, n_filters*8, kernel_size=2, stride=2)
        self.c6 = Conv2dBlock(n_filters*16, n_filters*8, batchnorm=batchnorm)
        self.up7 = nn.ConvTranspose2d(n_filters*8, n_filters*4, kernel_size=2, stride=2)
        self.c7 = Conv2dBlock(n_filters*8, n_filters*4, batchnorm=batchnorm)
        self.up8 = nn.ConvTranspose2d(n_filters*4, n_filters*2, kernel_size=2, stride=2)
        self.c8 = Conv2dBlock(n_filters*4, n_filters*2, batchnorm=batchnorm)
        self.up9 = nn.ConvTranspose2d(n_filters*2, n_filters, kernel_size=2, stride=2)
        self.c9 = Conv2dBlock(n_filters*2, n_filters, batchnorm=batchnorm)

        # Final output layer with 1x1 convolution
        self.outputs = nn.Conv2d(n_filters, n_classes, kernel_size=1)

    # Forward pass through the U-Net
    def forward(self, x):
        # Contracting path with max pooling
        c1 = self.c1(x)
        p1 = F.max_pool2d(c1, kernel_size=2, stride=2)

        c2 = self.c2(p1)
        p2 = F.max_pool2d(c2, kernel_size=2, stride=2)

        c3 = self.c3(p2)
        p3 = F.max_pool2d(c3, kernel_size=2, stride=2)

        c4 = self.c4(p3)
        p4 = F.max_pool2d(c4, kernel_size=2, stride=2)

        c5 = self.c5(p4)

        # Expansive path with transposed convolutions and skip connections
        u6 = self.up6(c5)
        u6 = F.interpolate(u6, size=c4.size()[2:], mode='bilinear', align_corners=False)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.c6(u6)

        u7 = self.up7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.c7(u7)

        u8 = self.up8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.c8(u8)

        u9 = self.up9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.c9(u9)

        # Final output layer
        outputs = self.outputs(c9)
        return outputs
