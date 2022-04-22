#!/usr/bin/env python
import sys

import numpy as np
import torch

from .model import FeatureGenerator, TraceGenerator, getparam, intp
from .normalizing_flows import NormalizingFlow
from .utils import random_normal_samples

import os

__ROOT__ = os.path.abspath(os.path.dirname(__file__))

def appinfo(scaleid):
    return np.load("%s/../data/appinfo_scale%d.npy" % (__ROOT__, scaleid + 1))    

def tracegen(scaleid, appid, nsample=20):
    ## Load metadata
    scale = "scale%d" % (scaleid + 1)
    appinfo = np.load("%s/../data/appinfo_%s.npy" % (__ROOT__, scale))
    nlen, nclass, mdim = getparam(scale)

    ## Feature Generatior
    fgen = FeatureGenerator(scale)
    fgen.load("%s/../data/fgen_%s_app%d.torch" % (__ROOT__, scale, appid))

    fgen.eval()
    features = (fgen.model.sample(random_normal_samples(nsample, dim=mdim))).detach().numpy()

    ## Trace Generatior
    tgen = TraceGenerator(scale)
    tgen.load("%s/../data/tgen_%s.torch" % (__ROOT__, scale))

    ## Prepare input for TG
    features_plus_app = np.zeros((len(features), len(appinfo)), dtype=np.float32)
    features_plus_app[:, appid] = 1.0
    features_plus_app = np.hstack((features, features_plus_app))
    features_plus_app = torch.tensor(features_plus_app)

    traces = tgen(features_plus_app)
    traces = traces.detach().cpu().numpy()

    return scale, appinfo[appid], traces
