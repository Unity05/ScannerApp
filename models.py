import torch
import torch.nn as nn
import numpy as np


class DenoisingAutoencoderSheetEdges(nn.Module):

    def __init__(self):
        super(DenoisingAutoencoderSheetEdges, self).__init__()
        self.hard_mode = False
        self.threshold = 0.75

        self.encoder = nn.Sequential(
            nn.ConvTranspose2d(1, 6, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 16, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 25, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(25, 50, kernel_size=5),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(50, 25, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(25, 16, kernel_size=5),   # transposed conv --> increases size (https://towardsdatascience.com/is-the-transposed-convolution-layer-and-convolution-layer-the-same-thing-8655b751c3a1)
            nn.ReLU(True),
            nn.Conv2d(16, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6, 1, kernel_size=5),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        if self.hard_mode:
            if torch.cuda.is_available():
                x = np.heaviside((x - self.threshold).cpu().detach(), 0).requires_grad_().cuda()
            else:
                x = np.heaviside((x-self.threshold).cpu().detach(), 0).requires_grad_()
        return x


class DenoisingAutoencoderLayoutParts(nn.Module):

    def __init__(self):
        super(DenoisingAutoencoderLayoutParts, self).__init__()

        self.encoder = nn.Sequential(
            nn.ConvTranspose2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 16, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 25, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(25, 50, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(50, 60, kernel_size=5),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(60, 50, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(50, 25, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(25, 16, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(16, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6, 3, kernel_size=5),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
