# -*- coding: utf-8 -*-
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq


def Bfx_lbp(I, R, options):

    if 'show' not in options:
        options['show'] = False

    if 'normalize' not in options:
        options['normalize'] = False

    if options['show']:
        print('--- extracting local binary patterns features...')

    if 'samples' not in options:
        options['samples'] = 8

    if 'integral' not in options:
        options['integral'] = 0

    if 'radius' not in options:
        options['radius'] = np.log(options['samples']) / np.log(2.0) - 1

    if 'semantic' not in options:
        options['semantic'] = 0

    if 'weight' not in options:
        options['weight'] = 0

    P = options['samples']
    R = options['radius']
    LBP = local_binary_pattern(I, P=P, R=2, method='nri_uniform')
    print(LBP.max())
    X = (np.histogram(LBP[2:-2, 2:-2].ravel(), 59)[0])[None]
    #X = itemfreq(LBP)
    if options['normalize']:
        X /= float(X.sum())

    #1 LBP(1,1)[8,u2]
    Xn = ['{0} LBP(1,{1})[{2},{3}]'.format(i, i, P, options['mappingtype']) for i in range(X.shape[1])]

    return X, Xn