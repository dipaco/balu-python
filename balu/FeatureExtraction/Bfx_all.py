# -*- coding: utf-8 -*-
import numpy as np


def Bfx_all(I, R=None, options=None):
    """X, Xn = Bfx_all(I,R,options)
    X, Xn = Bfx_all(I,options)

    Toolbox: Balu
       All pixels.

       X is the features vector, Xn is the list feature names (see Example to
       see how it works).


       Example:
          I = imread('testimg1.jpg')             # input image
          X, Xn = Bfx_all(I)                   # all pixels


    (c) D.Mery, PUC-DCC, 2012
    http://dmery.ing.puc.cl

    With collaboration from:
    Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    X = np.zeros((1, I.size))
    X[0, :] = I.ravel()
    Xn = X.size * ['']
    return X, Xn
