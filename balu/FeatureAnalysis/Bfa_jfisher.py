# -*- coding: utf-8 -*-
import numpy as np
from warnings import filterwarnings

def Bfa_jfisher(X, d, p=None):
    """ J = Bfa_jfisher(X, d, p)

     Toolbox: Balu
        Fisher objective function J.
        X features matrix. X(i,j) is the feature j of sample i.
        d vector that indicates the ideal classification of the samples
        p a priori probability of each class

     See also Bfs_sfs.

     (c) D.Mery, PUC-DCC, 2011
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
     """

    if len(d.shape) == 1:
        d = np.expand_dims(d, axis=1)

    n, M = X.shape
    dmin = d.min()

    d = d - dmin + 1

    N = int(d.max())

    if p is None:
        p = np.ones((N, 1)) / float(N)

    # Centroid of all samples
    Xm = np.expand_dims(np.mean(X, axis=0), axis=0).T

    L = np.zeros((N, 1))
    Cw = np.zeros((M, M))
    Cb = np.zeros((M, M))

    for k in range(N):
        ii, jj = np.where(d == k + 1)                           # indices from class k
        L[k] = ii.size                                          # number of samples of class k
        Xk = X[ii, :]                                           # samples of class k
        Xkm = np.expand_dims(np.mean(Xk, axis=0), axis=0).T     # centroid of class k
        Ck = np.cov(Xk, rowvar=False)                           # covariance of class k

        # within-class covariance
        Cw = Cw + p[k]*Ck

        # between-class covariance
        Cb = Cb + p[k]*np.dot(Xkm - Xm, (Xkm - Xm).T)


    # Fisher discriminant
    filterwarnings('ignore')
    #print Cw
    xxx, _, _, _ = np.linalg.lstsq(Cw, Cb)
    #xxx = np.linalg.pinv(Cw)
    #J = np.trace(np.linalg.solve(Cw, Cb))
    J = np.trace(xxx)
    return J
