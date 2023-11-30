import torch
import torch.nn as nn

## UNet Model
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters=16, dropout=0.1, batchnorm=True):
        super(UNet, self).__init__()

        # Contracting Path
        self.c1 = self.conv2d_block(in_channels, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
        self.p1 = nn.MaxPool2d(2)
        self.d1 = nn.Dropout2d(dropout)

        self.c2 = self.conv2d_block(n_filters * 1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
        self.p2 = nn.MaxPool2d(2)
        self.d2 = nn.Dropout2d(dropout)

        self.c3 = self.conv2d_block(n_filters * 2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
        self.p3 = nn.MaxPool2d(2)
        self.d3 = nn.Dropout2d(dropout)

        self.c4 = self.conv2d_block(n_filters * 4, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
        self.p4 = nn.MaxPool2d(2)
        self.d4 = nn.Dropout2d(dropout)

        self.c5 = self.conv2d_block(n_filters * 8, n_filters * 16, kernel_size=3, batchnorm=batchnorm)

        # Expansive Path
        self.u6 = nn.ConvTranspose2d(n_filters * 8, n_filters * 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.c6 = self.conv2d_block(n_filters * 16, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

        self.u7 = nn.ConvTranspose2d(n_filters * 4, n_filters * 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.c7 = self.conv2d_block(n_filters * 8, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

        self.u8 = nn.ConvTranspose2d(n_filters * 2, n_filters * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.c8 = self.conv2d_block(n_filters * 4, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

        self.u9 = nn.ConvTranspose2d(n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.c9 = self.conv2d_block(n_filters * 2, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

        self.outputs = nn.Conv2d(n_filters, out_channels, kernel_size=1, stride=1, padding=0)

    def conv2d_block(self, in_channels, out_channels, kernel_size=3, batchnorm=True):
        """Function to add 2D convolutional block with batch normalization and ReLU activation"""
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2))
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Contracting Path
        c1 = self.c1(x)
        p1 = self.p1(c1)
        d1 = self.d1(p1)

        c2 = self.c2(d1)
        p2 = self.p2(c2)
        d2 = self.d2(p2)

        c3 = self.c3(d2)
        p3 = self.p3(c3)
        d3 = self.d3(p3)

        c4 = self.c4(d3)
        p4 = self.p4(c4)
        d4 = self.d4(p4)

        c5 = self.c5(d4)

        # Expansive Path
        u6 = self.u6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.c6(u6)

        u7 = self.u7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.c7(u7)

        u8 = self.u8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.c8(u8)

        u9 = self.u9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.c9(u9)

        model = self.outputs(c9)

        return model

# Example usage:
# input_img = torch.randn(1, 3, 256, 256)  # Assuming input image size is 256x256
# model = UNet(in_channels=3, out_channels=1)
# output = model(input_img)
