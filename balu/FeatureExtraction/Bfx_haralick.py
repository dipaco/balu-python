# -*- coding: utf-8 -*-
import numpy as np


def Bfx_haralick(I, R=None, options={}):
    """ X, Xn = Bfx_haralick(I, R, options)
     X, Xn = Bfx_haralick(I, options)
    
     Toolbox: Balu
        Haralick texture features.
    
        X is a 28 elements vector with mean and range of mean and range of

         1 Angular Second Moment
         2 Contrast
         3 Correlacion
         4 Sum of squares
         5 Inverse Difference Moment
         6 Sum Average
         8 Sum Entropy
         7 Sum Variance
         9 Entropy
        10 Difference Variance
        11 Difference Entropy
        12,13 Information Measures of Correlation
        14 Maximal Correlation Coefficient

        Xn is the list of name features.

        I is the image. R is the binary image that indicates which pixels of I will be
        computed.
        options['dharalick'] is the distance in pixels used to compute the
        coocurrence matrix.
        options['show'] = True display results.

        Reference:
        Haralick (1979): Statistical and Structural Approaches to Texture,
        Proc. IEEE, 67(5):786-804

       Example 1: only one distance (3 pixels)
            from balu.ImagesAndData import balu_imageload
            from balu.ImageProcessing import Bim_segbalu
            from balu.FeatureExtraction import Bfx_haralick
            from balu.InputOutput import Bio_printfeatures

            options = {'dharalick': 3}              #3 pixels distance for coocurrence
            I = balu_imageload('testimg1.jpg')      #input image
            R, _, _ = Bim_segbalu(I)                #segmentation
            J = I[:, :, 1]                          #green channel
            X, Xn = Bfx_haralick(J, R, options)     #Haralick features
            Bio_printfeatures(X, Xn)

       Example 2: five distances (1,2,...5 pixels)
            from balu.ImagesAndData import balu_imageload
            from balu.ImageProcessing import Bim_segbalu
            from balu.FeatureExtraction import Bfx_haralick
            from balu.InputOutput import Bio_printfeatures

            options = {'dharalick': [1, 2, 3, 4, 5]}         #3 and 5 pixels distance for coocurrence
            I = balu_imageload('testimg1.jpg')      #input image
            R, _, _ = Bim_segbalu(I)                #segmentation
            J = I[:, :, 1]                          #green channel
            X, Xn = Bfx_haralick(J, R, options)     #Haralick features
            Bio_printfeatures(X, Xn)

       See also Bfx_gabor, Bfx_clp, Bfx_fourier, Bfx_dct, Bfx_lbp.

     (c) GRIMA-DCCUC, 2011
     http://grima.ing.puc.cl
    
     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """
    I = I.astype(float)

    if R is None:
        R = np.ones(I.shape)

    dseq = options['dharalick']

    if 'show' not in options:
        options['show'] = False

    if options['show']:
        print('--- extracting Haralick texture features...')

    if isinstance(dseq, (list, np.dtype)):
        dseq = np.array(dseq)
    else:
        dseq = np.array([dseq])

    m = dseq.size
    n = 28 * m

    X = np.zeros((1, n))
    Xn = n * [None]
    k = 0
    for i in range(m):
        d = dseq[i]

        Cd000 = Bcoocurrencematrix(I, R, d, 0) + Bcoocurrencematrix(I, R, -d, 0)
        Cd000 = Cd000 / np.sum(Cd000)
        Cd045 = Bcoocurrencematrix(I, R, d, -d) + Bcoocurrencematrix(I, R, -d, d)
        Cd045 = Cd045 / np.sum(Cd045)
        Cd090 = Bcoocurrencematrix(I, R, 0, d) + Bcoocurrencematrix(I, R, 0, -d)
        Cd090 = Cd090 / np.sum(Cd090)
        Cd135 = Bcoocurrencematrix(I, R, d, d) + Bcoocurrencematrix(I, R, -d, -d)
        Cd135 = Cd135 / np.sum(Cd135)

        TexMat = np.hstack((Bcoocurrencefeatures(Cd000), Bcoocurrencefeatures(Cd045), Bcoocurrencefeatures(Cd090), Bcoocurrencefeatures(Cd135)))
        X[0, i*28:(i+1)*28] = np.hstack((np.mean(TexMat, axis=1), np.max(np.abs(TexMat.T), axis=0)))

        for q in range(2):
            if q == 0:
                sq = 'mean '
            else:
                sq = 'range'

            for s in range(14):
                Xn[k] = 'Tx{0:2d},d{1:2d}({2})         '.format(s, d, sq)
                k += 1

    return X, Xn


def Bcoocurrencematrix(I, R, Io, Jo):
    """P = Bcoocurrencematrix(I, R, Io, Jo)

     Coocurrence matrix of the pixels of image I indicated by binary image R
     following the direction (Io,Jo).

     (c) D.Mery, PUC-DCC, Apr. 2008

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    V = np.floor(I / 32.0)
    N, M = I.shape
    Z1 = np.zeros((N+40, M+40))
    Z2 = Z1.copy()
    R1 = Z1.copy()
    R2 = R1.copy()
    Z1[14:N+14, 14:M+14] = V
    Z2[14+Io:N+14+Io, 14+Jo:M+14+Jo] = V
    R1[14:N+14, 14:M+14] = R
    R2[14+Io:N+14+Io, 14+Jo:M+14+Jo] = R
    ii, jj = np.where(np.logical_not(np.logical_and(R1, R2)))
    Z1[ii, jj] = -1
    Z2[ii, jj] = -1
    T1 = Z1.ravel()
    T2 = Z2.ravel()
    d = np.where(np.logical_and((T1 > -1), (T2 > -1)))[0]

    if d.size > 0:
        P = np.zeros((8, 8))
        T = np.vstack((T1[d], T2[d])).T
        I = np.lexsort((T2[d], T1[d]))
        X = T[I, :]
        ne = X.shape[0]
        i1 = np.where(np.logical_or(
            (np.insert(X[:, 0], 0, -1) - np.insert(X[:, 0], ne, -1)) != 0,
            (np.insert(X[:, 1], 0, -1) - np.insert(X[:, 1], ne, -1)) != 0
        ))[0]
        i2 = np.insert(i1[1:], i1.size - 1, -1)
        d = i2 - i1
        for i in range(d.size - 1):
            P[int(X[i1[i], 1]), int(X[i1[i], 0])] = d[i]

    else:
        P = -np.ones((8, 8))

    return P


def Bcoocurrencefeatures(P):
    """ Tx = Bcoocurrencefeatures(P)

     Haralick texture features calculated from coocurrence matrix P.

     (c) D.Mery, PUC-DCC, Apr. 2008

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    Pij = P.ravel()[:, None]
    Ng = 8
    pxi = np.sum(P, axis=1)[:, None]
    pyj = np.sum(P, axis=0)[:, None]

    ux = np.mean(pxi, axis=0)
    uy = np.mean(pyj, axis=0)

    sx = np.std(pxi, axis=0)
    sy = np.std(pyj, axis=0)

    pxy1 = np.zeros((2*Ng-1, 1))
    for k in range(1, 2*Ng):
        s = 0
        for i in range(Ng):
            for j in range(Ng):
                if i + j == k - 1:
                    s += P[i, j]

        pxy1[k - 1] = s

    pxy2 = np.zeros((Ng, 1))
    for k in range(Ng):
        s = 0
        for i in range(Ng):
            for j in range(Ng):
                if np.abs(i-j) == k:
                    s += P[i, j]

        pxy2[k] = s

    Q = np.zeros((Ng, Ng))
    pxi += 1e-20
    pyj += 1e-20

    for i in range(Ng):
        for j in range(Ng):
            s = 0
            for k in range(Ng):
                s += P[i, k] * P[j, k] / pxi[i] / pyj[k]

            Q[i, j] = s

    eigQ = np.linalg.eig(Q)[0]

    i, j = np.where(P >= 0)
    dif = i - j
    dif2 = dif * dif
    dif21 = dif2 + 1


    # 1 Angular Second Moment
    f1 = np.dot(Pij.T, Pij)[0][0]

    # 2 Contrast
    f2 = np.dot(np.arange(Ng)[None]**2, pxy2)[0][0]

    # 3 Correlacion
    f3 = (np.sum(i*j*np.squeeze(Pij)) - ux*uy*Ng**2)/sx/sy

    # 4 Sum of squares
    f4 = np.dot(dif2[None], Pij)[0][0]

    # 5 Inverse Difference Moment
    f5 = np.sum(Pij / dif21[:, None])

    # 6 Sum Average
    f6 = np.dot(np.arange(2, 2*Ng + 1)[None], pxy1)[0][0]

    # 8 Sum Entropy
    #f8 = -pxy1'*log(pxy1+1e-20);
    f8 = np.dot(-pxy1.T, np.log(pxy1 + 1e-20))[0][0]

    # 7 Sum Variance
    if8 = np.arange(2, 2*Ng + 1)[None] - f8
    f7 = np.dot(if8, pxy1)[0][0]

    # 9 Entropy
    f9 = np.dot(-Pij.T, np.log(Pij+1e-20))[0][0]

    # 10 Difference Variance
    f10 = np.var(pxy2, axis=0)

    # 11 Difference Entropy
    f11 = np.dot(-pxy2.T, np.log(pxy2+1e-20))[0][0]

    # 12,13 Information Measures of Correlation
    HXY = f9
    pxipyj = pxi[i] * pyj[j]
    HXY1 = np.dot(-Pij.T, np.log(pxipyj + 1e-20))[0][0]
    HXY2 = np.dot(-pxipyj.T, np.log(pxipyj+1e-20))
    HX = np.dot(-pxi.T, np.log(pxi + 1e-20))[0][0]
    HY = np.dot(-pyj.T, np.log(pyj + 1e-20))[0][0]
    f12 = ((HXY-HXY1) / max(HX, HY))
    f13 = (1 - np.exp(-2 * (HXY2-HXY)))

    # 14 Maximal Corrleation Coefficient
    f14 = eigQ[1]

    Tx = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14]]).T

    return Tx
