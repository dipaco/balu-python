# -*- coding: utf-8 -*-
import numpy as np


def Bim_inthist(I, b=256):
    """H = Bim_inthist(I, b)


     Toolbox: Balu
        Integral histogram.

        Input data:
           I grayvalue image (only for positive values)
           b number of bins

        Output:
           H Integral histogram (size = NxMxb, where [N,M] = size(I))

        Example:
            import numpy as np
            from balu.ImagesAndData import balu_imageload
            from balu.FeatureExtraction import Bfx_lbp
            from balu.ImageProcessing import Bim_inthist, Bim_inthistread

            I = balu_imageload('rice.png')
            H = Bim_inthist(I, 256)
            i1 = 70
            j1 = 50
            i2 = 130
            j2 = 190
            K = I[i1:i2+1, j1:j2+1]         #i2 and j2 are included in the rectangle
            t = np.histogram(K.ravel(), bins=np.arange(-0.5, 256))[0]
            h = Bim_inthistread(H, i1, j1, i2, j2)
            print('h and t are equal: {0}'.format(np.sum(h-t) == 0))

        See also Bim_inthistread.

     (c) D.Mery, PUC-DCC, 2012
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """
    N, M = I.shape
    J = I.astype(int)
    s = np.log(N * M)/np.log(2.0)
    H = np.zeros((N, M, b))

    B = J[0, 0]
    H[0, 0, B] = 1
    i = 0
    for j in range(1, M):
        B = J[0, j]
        H[i, j, :] = H[i, j-1, :]
        H[i, j, B] += 1

    for i in range(1, N):
        B = J[i, 0]
        H[i, 0, :] = H[i-1, 0, :]
        H[i, 0, B] += 1
        for j in range(1, M):
            B = J[i, j]
            H[i, j, :] = H[i, j-1, :] + H[i-1, j, :] - H[i-1, j-1, :]
            H[i, j, B] += 1

    return H
