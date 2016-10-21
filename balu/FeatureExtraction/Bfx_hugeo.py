# -*- coding: utf-8 -*-
import numpy as np


def Bfx_hugeo(R, options={}):
    """[X, Xn] = Bfx_hugeo(R)
     [X, Xn] = Bfx_hugeo(R, options)

     Toolbox: Balu

     Extract the seven Hu moments from binary image R.

     options.show = 1 display mesagges.

     X is a 7 elements vector:
     X(i): Hu - moment i for i=1, ..., 7.
     Xn is the list of feature names.

     Reference:
     Hu, M - K.: "Visual Pattern Recognition by Moment Invariants",
     IRE Trans.Info.Theory IT - 8:179 - 187: 1962.

    Example:
    import numpy as np
    from balu.FeatureExtraction import Bfx_hugeo
    from balu.ImagesAndData import balu_imageload
    from balu.ImageProcessing import Bim_segbalu
    from skimage.morphology import label
    from matplotlib.pyplot import imshow, show

    I = balu_imageload('testimg3.jpg')              # input image
    R = Bim_segbalu(I)                              # segmentation
    L, n = label(R, neighbors=8, return_num=True)   # regions
    imshow(L, cmap='gray')
    X = None
    for i in range(1, n + 1):
        Xi, Xn = Bfx_hugeo(L == i)                  # Hu moments
        if X is None:
            X = Xi
        else:
            X = np.vstack((X, Xi))

    print X
    show()

     See also Bfx_basicgeo, Bfx_gupta, Bfx_fitellipse, Bfx_flusser.

    (c)
    D.Mery, PUC - DCC, 2010
    http: // dmery.ing.puc.cl

    With collaboration from:
    Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    if 'show' not in options:
        options['show'] = False

    if options['show']:
        print('--- extracting Hu moments...')

    Ireg, Jreg = np.where(R.astype('uint8') == 1)         # pixels in the region
    Ireg = np.expand_dims(Ireg, axis=1)
    Jreg = np.expand_dims(Jreg, axis=1)

    i_m = np.mean(Ireg)
    j_m = np.mean(Jreg)
    A = Ireg.size
    I0 = np.ones((A, 1))
    J0 = np.ones((A, 1))
    I1 = Ireg - i_m * np.ones((A, 1))
    J1 = Jreg - j_m * np.ones((A, 1))
    I2 = I1 * I1
    J2 = J1 * J1
    I3 = I2 * I1
    J3 = J2 * J1

    # Central moments
    u00 = A             # u00 = m00 = (I0 '*J0)
    u002 = u00 * u00
    u0025 = u00 ** 2.5

    # u0015 = u00 ** 1.5 not used
    n02 = np.dot(I0.T, J2)/u002
    n20 = np.dot(I2.T, J0)/u002
    n11 = np.dot(I1.T, J1)/u002
    n12 = np.dot(I1.T, J2)/u0025
    n21 = np.dot(I2.T, J1)/u0025
    n03 = np.dot(I0.T, J3)/u0025
    n30 = np.dot(I3.T, J0)/u0025

    f1 = n20 + n02
    f2 = (n20 - n02) ** 2 + 4 * n11 ** 2
    f3 = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    f4 = (n30 + n12) ** 2 + (n21 + n03) ** 2
    f5 = (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) + (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    f6 = (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4 * n11 * (n30 + n12) * (n21 + n03)
    f7 = (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) - (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)

    X = np.zeros((1, 7))
    X[0, 0] = f1
    X[0, 1] = f2
    X[0, 2] = f3
    X[0, 3] = f4
    X[0, 4] = f5
    X[0, 5] = f6
    X[0, 6] = f7

    Xn = ['Hu-moment 1',
          'Hu-moment 2',
          'Hu-moment 3',
          'Hu-moment 4',
          'Hu-moment 5',
          'Hu-moment 6',
          'Hu-moment 7']

    return X, Xn
