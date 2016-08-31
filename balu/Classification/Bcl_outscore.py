# -*- coding: utf-8 -*-
import numpy as np
from balu.FeatureTransformation import Bft_sigmoid


def Bcl_outscore(ds, score, options):

    outp = 0
    if 'output' in options:
        outp = options['output']

    if outp == 1:
        ds = score
    elif outp == 2:
        ds = np.concatenate((ds.ravel(), score.ravel()))
    elif outp == 3:
        ds = np.concatenate((ds.ravel(), (Bft_sigmoid(score, options['param']) - 0.5).ravel() * 2))

    return ds
