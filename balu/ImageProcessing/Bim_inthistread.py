# -*- coding: utf-8 -*-
import numpy as np


def Bim_inthistread(H, x1, y1, x2, y2, n=1):
    """ h = Bim_inthistread(H, x1, y1, x2, y2, n)

     Toolbox: Balu
        Histogram of a part of an image using integral histograms.

        Input data:
           H integral histogram.
           (i1,j1,i2,j2) rectangle of the image. i2 and j2 are including in the rectangle.
           n number of subdivisions (in a regular grid) of the rectangle specified by x1, y1, x2 and y2

        Output:
           h histogram

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

        See also Bim_inthist.

     (c) D.Mery, PUC-DCC, 2012
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """
    if len(H.shape) > 2:
        B = H.shape[2]
    else:
        B = 1

    if n == 1:
        h = _suminthist(H, x1, y1, x2, y2, B)
    else:
        #[x1 y1 x2 y2]
        N = H.shape[0]
        M = H.shape[1]
        t = 0
        h = np.zeros(n**2*B)
        dx = (x2+1-x1)/float(n)
        dy = (y2+1-y1)/float(n)
        xx1 = x1
        for i in range(0, n):
            xx2 = min(np.round(xx1+dx)-1, N)
            yy1 = y1
            for j in range(0, n):
                t += 1
                yy2 = min(np.round(yy1+dy)-1, M)
                # [i j xx1 yy1 xx2 yy2]
                h[((t-1)*B) + np.arange(B)] = _suminthist(H, int(xx1), int(yy1), int(xx2), int(yy2), B)
                yy1 = yy2+1

            xx1 = xx2+1
    return h

    #print('****')


def _suminthist(H, x1, y1, x2, y2, B):
    h = np.zeros(B)
    t = np.zeros(B)
    # [0 0 x1 y1 x2 y2]
    # size(H)

    h[:] = H[x2, y2, :]

    if x1 > 0 and y1 > 0:
        t[:] = H[x1-1, y1-1, :]
        h[:] = h[:] + t[:]

    if x1 > 0:
        t[:] = H[x1-1, y2, :]
        h[:] = h[:]-t[:]

    if y1 > 0:
        t[:] = H[x2, y1-1, :]
        h[:] = h[:] - t[:]
    return h
