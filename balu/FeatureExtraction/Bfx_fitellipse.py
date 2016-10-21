# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage.morphology import binary_erosion


def Bfx_fitellipse(R, options=None):
    """ X, Xn = Bfx_fitellipse(R, options)
     X, Xn = Bfx_fitellipse(R)

     Toolbox: Balu

        Fit ellipse for the boundary of a binary image R.

        options.show = 1 display mesagges.

        X is a 6 elements vector:
          X(1): Ellipse-centre i direction
          X(2): Ellipse-centre j direction
          X(3): Ellipse-minor axis
          X(4): Ellipse-major axis
          X(5): Ellipse-orientation
          X(6): Ellipse-eccentricity
          X(7): Ellipse-area

          Xn is the list of feature names.

          Xn is the list of feature names.

       Reference:
          Fitzgibbon, A.; Pilu, M. & Fisher, R.B. (1999): Direct Least Square
          Fitting Ellipses, IEEE Trans. Pattern Analysis and Machine
          Intelligence, 21(5): 476-480.

       Example:
            from balu.ImagesAndData import balu_imageload
            from balu.ImageProcessing import Bim_segbalu
            from balu.FeatureExtraction import Bfx_fitellipse
            from balu.InputOutput import Bio_printfeatures

            I = balu_imageload('testimg1.jpg')              # input image
            R, E, J = Bim_segbalu(I)                        # segmentation
            X, Xn = Bfx_fitellipse(R)                       # ellipse features
            Bio_printfeatures(X, Xn)

       See also Bfx_basicgeo, Bfx_hugeo, Bfx_gupta, Bfx_flusser.

     (c) D.Mery, PUC-DCC, 2010
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    E = R - binary_erosion(R)
    Y, X = np.where(E == 1)                 # pixel of perimeter in (i,j)
    if X.size > 5:

        if options is None:
            options = {'show': False}

        if options['show']:
            print('--- extracting ellipse features...')

        # normalize data
        mx = np.mean(X)
        my = np.mean(Y)
        sx = (X.max() - X.min()) / 2.0
        sy = (Y.max() - Y.min()) / 2.0
        x = (X - mx) / sx
        y = (Y - my) / sy

        # Build design matrix
        D = np.vstack((x*x,  x*y,  y*y,  x,  y, np.ones(x.size))).T

        U, S, V = np.linalg.svd(D)
        A = V.T[:, 5]

        # unnormalize
        a = np.array([[
            A[0]*sy*sy,
            A[1]*sx*sy,
            A[2]*sx*sx,
            -2*A[0]*sy*sy*mx - A[1]*sx*sy*my + A[3]*sx*sy*sy,
            -A[1]*sx*sy*mx - 2*A[2]*sx*sx*my + A[4]*sx*sx*sy,
            A[0]*sy*sy*mx*mx + A[1]*sx*sy*mx*my + A[2]*sx*sx*my*my - A[3]*sx*sy*sy*mx - A[4]*sx*sx*sy*my + A[5]*sx*sx*sy*sy]]).T

        a = a / a[5, 0]

        # get ellipse orientation
        alpha = np.arctan2(a[1, 0], a[0, 0] - a[2, 0]) / 2.0

        # get scaled major/minor axes
        ct = np.cos(alpha)
        st = np.sin(alpha)
        ap = a[0]*ct*ct + a[1]*ct*st + a[2]*st*st
        cp = a[0]*st*st - a[1]*ct*st + a[2]*ct*ct

        # get translations
        T = np.zeros((2, 2))
        T[0, 0] = a[0, 0]
        T[0, 1] = a[1, 0] / 2.0
        T[1, 0] = a[1, 0] / 2.0
        T[1, 1] = a[2, 0]
        p = np.array([a[3], a[4]])
        mc = np.dot(-np.linalg.pinv(2*T), p)

        # get scale factor
        val = np.dot(mc.T, np.dot(T, mc))
        scale = np.abs(1.0 / (val - a[5]))

        # get major/minor axis radii
        ae = 1.0 / np.sqrt(scale * np.abs(ap))
        be = 1.0 / np.sqrt(scale * np.abs(cp))
        ecc = ae / be  # eccentricity
        ar = np.pi * ae * be

        X = np.array([[mc[0, 0], mc[1, 0], ae, be, alpha, ecc, ar]])
    else:
        X = np.array([[0., 0., 0., 0., 0., 0., 0.]])
        print('Warning: Bfx_fitellipse does not have enough points to fit')

    Xn = [
        'Ellipse-centre i [px]   ',
        'Ellipse-centre j [px]   ',
        'Ellipse-minor ax [px]   ',
        'Ellipse-major ax [px]   ',
        'Ellipse-orient [rad]    ',
        'Ellipse-eccentricity    ',
        'Ellipse-area [px]       '
        ]
    return X, Xn
