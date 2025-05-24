# MIT License

# Original work Copyright (c) 2018 Joris (https://github.com/jvanvugt/pytorch-unet)
# Modified work Copyright (C) 2022 Canon Medical Systems Corporation.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt

from swish import CustomSwish
from ws_conv import WNConv2d


def get_groups(channels: int) -> int:
    """
    :param channels:
    :return: return a suitable parameter for number of groups in GroupNormalisation'.
    """
    divisors = []
    for i in range(1, int(sqrt(channels)) + 1):
        if channels % i == 0:
            divisors.append(i)
            other = channels // i
            if i != other:
                divisors.append(other)
    return sorted(divisors)[len(divisors) // 2]

# Global Encoder
class GlobalEncoder(nn.Module):
    def __init__(self, in_channels=4, latent_dim=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),   # 64x64
            CustomSwish(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),           # 32x32
            CustomSwish(),
            nn.AdaptiveAvgPool2d(1),                              # 1x1
            nn.Flatten(),
            nn.Linear(128, latent_dim),
            CustomSwish()
        )

    def forward(self, x):
        return self.enc(x)



class UNet(nn.Module):
    def __init__(
        self,
        in_channels=4,
        n_classes=4,
        depth=3,
        wf=6,
        padding=True,
        norm="group",
        up_mode='upconv',
        latent_dim=128
    ):
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        self.latent_dim = latent_dim

        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, norm=norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, norm=norm, latent_dim=latent_dim)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        # Integrate global encoder
        self.global_encoder = GlobalEncoder(in_channels, latent_dim)

    def forward_down(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            blocks.append(x)
            if i != len(self.down_path) - 1:
                x = F.avg_pool2d(x, 2)
        return x, blocks

    def forward_up_without_last(self, x, blocks, latent_vec):  # latent_vec added
        for i, up in enumerate(self.up_path):
            skip = blocks[-i - 2]
            x = up(x, skip, latent_vec)  # latent_vec passed here
        return x

    def forward(self, x):
        latent_vec = self.global_encoder(x)  # Encode global latent (NEW)
        x, blocks = self.forward_down(x)
        x = self.forward_up_without_last(x, blocks, latent_vec)  # latent_vec passed here
        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, norm="group", kernel_size=3):
        super(UNetConvBlock, self).__init__()
        block = []
        if padding:
            block.append(nn.ReflectionPad2d(1))

        block.append(WNConv2d(in_size, out_size, kernel_size=kernel_size))
        block.append(CustomSwish())

        if norm == "batch":
            block.append(nn.BatchNorm2d(out_size))
        elif norm == "group":
            block.append(nn.GroupNorm(get_groups(out_size), out_size))

        if padding:
            block.append(nn.ReflectionPad2d(1))

        block.append(WNConv2d(out_size, out_size, kernel_size=kernel_size))
        block.append(CustomSwish())

        if norm == "batch":
            block.append(nn.BatchNorm2d(out_size))
        elif norm == "group":
            block.append(nn.GroupNorm(get_groups(out_size), out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, norm="group", latent_dim=128):
        super().__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, norm=norm)

        # FiLM conditioning layers 
        self.gamma = nn.Linear(latent_dim, out_size)
        self.beta = nn.Linear(latent_dim, out_size)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, x, bridge, latent_vec):  # latent_vec added
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        
        out = self.conv_block(out)

        # FiLM conditioning
        gamma = self.gamma(latent_vec).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(latent_vec).unsqueeze(-1).unsqueeze(-1)
        out = gamma * out + beta

        return out
