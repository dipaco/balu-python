# -*- coding: utf-8 -*-
import numpy as np
from skimage.morphology import label


def Bim_segmowgli(I, a, b, c):
    if len(I.shape) > 2:
        I = I[:, :, 1] > 128
    else:
        I = I > 128

    L = label(I)

    return L, L.max()
