# -*- coding: utf-8 -*-
import numpy as np


def Bev_confusion(d, ds, nn=None):
    """ function [T,p] = Bev_confusion(d,ds,nn);

     Toolbox: Balu
        Confusion Matrix and Performance of a classification

        d is the ideal classification (vector Nx1 with N samples)
        ds is the classified data (vector Nx1)
        T is the confusion matrix (nxn) for n classes
        T(i,j) indicates the number of samples i classified
        as j.
        p is the performance, diagonal sum divided by N
        the classes should be labeled as 1, 2, ... n, but
        if nn = [i j] is given it indicates that the lowest
        class is min(nn) and the highest class is max(nn).
        Thus, n = max(nn)-min(nn)+1 and the size of T is nxn.

     D.Mery, PUC-DCC, 2010
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    if nn is not None:
        n1 = np.max(nn).astype(int)
        n0 = np.min(nn).astype(int)
    else:
        n1 = np.amax(np.concatenate((d.ravel(), ds.ravel()))).astype(int)
        n0 = np.amin(np.concatenate((d.ravel(), ds.ravel()))).astype(int)

    n = n1 - n0 + 1
    T = np.zeros((n, n))
    for i in range(n0, n1 + 1):
        for j in range(n0, n1 + 1):
            kd = (d == i).astype('uint8')
            kds = (ds == j).astype('uint8')
            T[i - n0, j - n0] = np.sum(kd.ravel() * kds.ravel())

    p = np.trace(T) / np.sum(T)

    return T, p