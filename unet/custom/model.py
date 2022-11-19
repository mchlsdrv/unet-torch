import logging
import pathlib

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

__author__ = 'mchlsdrv@gmail.com'


class UNet(nn.Module):
    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv2d = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.conv2d(x)

    def __init__(self, in_channels, out_channels, n_features=(64, 128, 256, 512), output_dir: pathlib.Path = None, logger: logging.Logger = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_features = n_features
        self.output_dir = output_dir
        self.logger = logger

        self.down_path_convs = nn.ModuleList()
        self.up_path_convs = nn.ModuleList()
        self.pooling_lyr = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck_lyr = None
        self.mask_pred_lyr = None

        self.build()

    # Build
    def build(self):
        # - Down convs
        in_channels = self.in_channels
        for conv_n_features in self.n_features:
            self.down_path_convs.append(
                self.DoubleConv(
                    in_channels=in_channels,
                    out_channels=conv_n_features
                )
            )
            in_channels = conv_n_features

        # - Up convs
        for deconv_n_features in reversed(self.n_features):
            self.up_path_convs.append(
                nn.ConvTranspose2d(
                    2 * deconv_n_features,
                    deconv_n_features,
                    kernel_size=2,
                    stride=2
                )
            )
            self.up_path_convs.append(
                self.DoubleConv(
                    in_channels=2 * deconv_n_features,
                    out_channels=deconv_n_features
                )
            )

        # - Bottleneck
        self.bottleneck_lyr = self.DoubleConv(
            in_channels=self.n_features[-1],
            out_channels=2 * self.n_features[-1]
        )

        self.mask_pred_lyr = nn.Conv2d(
            in_channels=self.n_features[0],
            out_channels=self.out_channels,
            kernel_size=1
        )

    def forward(self, x):
        skip_connections = list()

        for dwn_pth_conv in self.down_path_convs:
            x = dwn_pth_conv(x)
            skip_connections.append(x)
            x = self.pooling_lyr(x)

        x = self.bottleneck_lyr(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.up_path_convs), 2):
            x = self.up_path_convs[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.up_path_convs[idx + 1](concat_skip)

        return self.mask_pred_lyr(x)


def test1():
    x = torch.randn((3, 1, 160, 160))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape


def test2():
    x = torch.randn((3, 1, 161, 161))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape
