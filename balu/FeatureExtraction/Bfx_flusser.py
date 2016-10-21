# -*- coding: utf-8 -*-
import numpy as np


def Bfx_flusser(R, options={}):
    """ X, Xn = Bfx_flusser(R, options)
     X, Xn = Bfx_flusser(R)

     Toolbox: Balu

        Extract the four Flusser moments from binary image R.

        options.show = 1 display mesagges.

        X is a 4 elements vector:
          X(i): Flusser-moment i for i=1,...,4.
        Xn is the list of feature names.

        Reference:
        Sonka et al. (1998): Image Processing, Analysis, and Machine Vision,
        PWS Publishing. Pacific Grove, Ca, 2nd Edition.

        Example:
            from balu.ImagesAndData import balu_imageload
            from balu.ImageProcessing import Bim_segbalu
            from skimage.morphology import label
            from balu.FeatureExtraction import Bfx_flusser

            I = balu_imageload('testimg3.jpg')           # input image
            R, _, _ = Bim_segbalu(I)                           # segmentation
            L = label(R)                                 # regions
            n = int(L.max())
            imshow(L)
            show()
            X = np.array([])
            for i in range(n):
                Xi, Xn = Bfx_flusser(L == i + 1)         # Flusser moments
                if X.size == 0:
                    X = Xi
                else:
                    X = np.vstack((X, Xi))

            print X

       See also Bfx_standard, Bfx_hugeo, Bfx_fitellipse, Bfx_gupta.

     (c) D.Mery, PUC-DCC, 2010
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    if 'show' not in options:
        options['show'] = False

    if options['show']:
        print('--- extracting Flusser moments...')

    Ireg, Jreg = np.where(R == 1)           # pixels in the region

    i_m = np.mean(Ireg)
    j_m = np.mean(Jreg)
    A = Ireg.size
    I0 = np.ones((A, 1))
    J0 = np.ones((A, 1))
    I1 = Ireg[:, None] - i_m * np.ones((A, 1))
    J1 = Jreg[:, None] - j_m * np.ones((A, 1))
    I2 = I1 * I1
    J2 = J1 * J1
    I3 = I2 * I1
    J3 = J2 * J1

    # Central moments
    u00 = np.squeeze(np.dot(I0.T, J0))
    # u01 = np.dot(I0.T, J1); not used
    u02 = np.squeeze(np.dot(I0.T, J2))
    u03 = np.squeeze(np.dot(I0.T, J3))
    # u10 = np.dot(I1.T, J0); not used
    u20 = np.squeeze(np.dot(I2.T, J0))
    u30 = np.squeeze(np.dot(I3.T, J0))
    u11 = np.squeeze(np.dot(I1.T, J1))
    u12 = np.squeeze(np.dot(I1.T, J2))
    u21 = np.squeeze(np.dot(I2.T, J1))

    II1 = (u20*u02-u11**2)/u00**4
    II2 = (u30**2*u03**2-6*u30*u21*u12*u03+4*u30*u12**3+4*u21**3*u03-3*u21**2*u12**2)/u00**10
    II3 = (u20*(u21*u03-u12**2)-u11*(u30*u03-u21*u12)+u02*(u30*u12-u21**2))/u00**7
    II4 = (u20**3*u03**2-6*u20**2*u11*u12*u03-6*u20**2*u02*u21*u03+9*u20**2*u02*u12**2 + 12*u20*u11**2*u21*u03+6*u20*u11*u02*u30*u03-18*u20*u11*u02*u21*u12-8*u11**3*u30*u03- 6*u20*u02**2*u30*u12+9*u20*u02**2*u21+12*u11**2*u02*u30*u12-6*u11*u02**2*u30*u21+u02**3*u30**2)/u00**11

    X  = np.array([[II1, II2, II3, II4]])

    Xn = [ 'Flusser-moment 1        ',
           'Flusser-moment 2        ',
           'Flusser-moment 3        ',
           'Flusser-moment 4        ']
    return X, Xn
