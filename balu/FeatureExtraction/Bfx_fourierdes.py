# -*- coding: utf-8 -*-
import numpy as np
from skimage.measure import find_contours


def Bfx_fourierdes(R, options={}):
    """ X, Xn = Bfx_fourierdes(R, options)

     Toolbox: Balu
        Computes the Fourier descriptors of a binary image R.

        options['show'] = True display messages.
        options['Nfourierdes'] number of descriptors.

        X is the feature vector
        Xn is the list of feature names.

        Reference:
        Zahn, C; Roskies, R.: Fourier Descriptors for Plane
        Closed Curves, IEEE Trans on Computers, C21(3):269-281, 1972

       Example:
            from balu.ImagesAndData import balu_imageload
            from balu.FeatureExtraction import Bfx_fourierdes
            from balu.InputOutput import Bio_printfeatures
            from balu.ImageProcessing import Bim_segbalu

            I = balu_imageload('testimg1.jpg')      # input image
            R = Bim_segbalu(I)                      # segmentation
            X, Xn = Bfx_fourierdes(R)               # Fourier descriptors
            Bio_printfeatures(X, Xn)

       See also Bfx_fitellipse, Bfx_hugeo, Bfx_gupta, Bfx_flusser.

     (c) D.Mery, PUC-DCC, 2010
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    if len(options) == 0:
        options['show'] = False
        options['Nfourierdes'] = 16

    if options['show']:
        print('--- extracting Fourier descriptors...')

    N = options['Nfourierdes']

    jj = 1j

    B = find_contours(R, 0.9, positive_orientation='high')
    g = B[0]

    V = g[:, 1] + jj * g[:, 0]
    m = g.shape[0]

    r = np.zeros((m, 1)).astype(complex)
    phi = np.zeros((m, 1))
    dphi = np.zeros((m, 1))
    l = np.zeros((m, 1))
    dl = np.zeros((m, 1))

    r[0] = V[0] - V[m - 1]
    for i in range(1, m):
        r[i] = V[i] - V[i - 1]

    for i in range(m):
        dl[i] = np.abs(r[i])
        phi[i] = np.angle(r[i])

    for i in range(m - 1):
        dphi[i] = np.mod(phi[i + 1] - phi[i] + np.pi, 2 * np.pi) - np.pi

    dphi[m - 1] = np.mod(phi[0] - phi[m - 1] + np.pi, 2 * np.pi) - np.pi

    for k in range(m):
        l[k] = 0
        for i in range(k + 1):
            l[k] = l[k] + dl[i]

    L = l[m - 1]

    A = np.zeros((N, 1))

    for n in range(1, N + 1):
        an = 0
        bn = 0
        for k in range(m):
            an = an + dphi[k] * np.sin(2 * np.pi * n * l[k] / L)
            bn = bn + dphi[k] * np.cos(2 * np.pi * n * l[k] / L)

        an = -an / n / np.pi
        bn = bn / n / np.pi
        imagi = an + jj * bn
        A[n - 1] = np.abs(imagi)

    X = A.T
    Xn = N * ['']
    for i in range(N):
        Xn[i] = 'Fourier-des {0}'.format(i)

    return X, Xn
