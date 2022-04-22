import torch
import numpy as np
from scipy.interpolate import interp1d

from .normalizing_flows import NormalizingFlow


def intp(lab, n=40):
    vf1 = np.vectorize(lambda x: max(x, 0.0))
    vf2 = np.vectorize(lambda x: min(x, 1.0))

    lab = vf1(lab)
    lab = vf2(lab)

    x = np.linspace(0, 1, 20)
    intp = interp1d(x, lab, kind="slinear")

    xi = np.linspace(x.min(), x.max(), n)
    yi = intp(xi)

    return (xi, yi)


def getparam(scale):
    if scale == "scale1":
        nlen, nclass, mdim = 26, 10, 75 + 7
    if scale == "scale2":
        nlen, nclass, mdim = 220, 10, 60 + 7
    if scale == "scale3":
        nlen, nclass, mdim = 417, 7, 25 + 7

    return nlen, nclass, mdim


class ResNet_block(torch.nn.Module):
    def __init__(self, n, act=torch.nn.LeakyReLU()):
        super().__init__()
        self.module = torch.nn.Sequential(
            torch.nn.Linear(n, n), torch.nn.LeakyReLU(), torch.nn.Linear(n, n),
        )
        self.act = act

    def forward(self, inputs):
        x = self.module(inputs)
        return self.act(x + inputs)


class TraceGenerator(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()

        nlen, nclass, mdim = getparam(scale)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(nclass + mdim, 128),
            # 32 filters in and out, no max pooling so the shapes can be added
            ResNet_block(128),
            ResNet_block(128),
            torch.nn.Linear(128, 64),
            ResNet_block(64),
            ResNet_block(64),
            torch.nn.Linear(64, 32),
            ResNet_block(32),
            ResNet_block(32),
            torch.nn.Linear(32, 20),
        )

    def forward(self, x):
        return self.model(x)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))


class FeatureGenerator(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()

        nlen, nclass, mdim = getparam(scale)
        self.model = NormalizingFlow(mdim, 10)

    def forward(self, x):
        return self.model(x)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
