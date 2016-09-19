# -*- coding: utf-8 -*-


def Bim_segbalu(I):
    if len(I.shape) > 2:
        return I[:, :, 1] > 128
    else:
        return I > 128
