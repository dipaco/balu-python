# -*- coding: utf-8 -*-
import numpy as np


def Bfa_sp100(X, d):
    """ Sp = Bfa_sp100(X, d)

     Toolbox: Balu
        Especificty at Sensibility = 100%.
        X features matrix. X(i,j) is the feature j of sample i.
        d vector that indicates the ideal classification of the samples

     See also Bfs_sfs, Bfa_fisher

     (c) D.Mery, PUC-DCC, 2011
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    if len(d.shape) == 1:
        d = np.expand_dims(d, axis=1)

    N, m = X.shape

    d = d - d.min()
    if d.max() > 1:
        print('Bfa_sp100 works for only two classes')
        exit()

    d1, d2 = np.where(d == 1)

    minz = np.min(X[d1, :], axis=0)
    maxz = np.max(X[d1, :], axis=0)
    z1 = np.zeros(X.shape)
    for p in range(m):
        ii = np.where(np.logical_and(X[:, p] >= minz[p], X[:, p] <= maxz[p]))[0]
        z1[ii, p] = np.ones((ii.size))

    if m > 1:
       drs = np.sum(z1, axis=1)
       ii = np.where(drs == X.shape[1])[0]
       dr = np.zeros(d.shape)
       dr[ii] = np.ones((ii.size, 1))
    else:
       dr = z1


    TP = np.sum(dr*d, axis=0)
    FP = np.sum(dr, axis=0) - TP

    TN = np.sum((1 - dr) * (1 - d), axis=0)
    Sp = (TN/(FP+TN))[0]
    return Sp
