# -*- coding: utf-8 -*-


def Bim_maxmin(X):
    X = X.astype(float)
    Xmax = X.max()
    Xmin = X.min()
    Y = (X - Xmin) / (Xmax - Xmin)
    return Y