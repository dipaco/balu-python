# -*- coding: utf-8 -*-
import numpy as np


def Bft_norm(X, normtype):
    """ Xnew, a, b = Bft_norm(X, normtype)

     Toolbox: Balu

        Normalization of features X.
        normtype = 1 for variance = 1 and mean = 0
        normtype = 0 for max = 1, min = 0

        Xnew = a*X + b

     Example:
        from balu.ImagesAndData import balu_load
        from matplotlib.pyplot import figure
        from balu.InputOutput import Bio_plotfeatures
        from balu.FeatureTransformation import Bft_norm

        data = balu_load('datareal')
        f, d = data['f'], data['d']
        X = f[:, 0:2]
        figure(1); Bio_plotfeatures(X, d)
        Xnew, a, b = Bft_norm(X, 0)
        figure(2); Bio_plotfeatures(Xnew, d)

     (c) Grima, PUC-DCC, 2010: D. Mery, E.Cortazar
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    N, M = X.shape
    if normtype == 1:
        mf = np.mean(X, axis=0)
        sf = np.std(X, axis=0)
        a = np.ones((1, M)) / sf
        b = -mf / sf
    else:
        mi = np.min(X, axis=0)
        ma = np.max(X, axis=0)
        md = ma - mi + (ma == mi).astype(int)
        a = np.ones((1, M)) / md
        b = -mi / md

    Xnew = X * np.dot(np.ones((N, 1)), a) + np.ones((N, 1)) * b

    return Xnew, a, b