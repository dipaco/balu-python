# -*- coding: utf-8 -*-
import numpy as np


def Bds_nostratify(X, d, s):
    """ X1, d1, X2, d2 = Bds_nostratify(X, d, s)

     Toolbox: Balu

        Data Sampling without Stratification (without replacement)

        input: (X,d) means features and ideal classification
        Bnostratify takes randomily a portion s (s between 0 and 1) of the
        whole that without considering the portion of each class
        from (X,d) to build (X1,d1). The samples not used in (X1,d1) are
        stored in (X2,d2).

     D.Mery, PUC-DCC, 2010
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
     """
    if len(d.shape) < 2:
        d = d[:, None]

    N = X.shape[0]
    rn = np.random.rand(N)
    j = np.argsort(rn)
    Xr = X[j, :]
    dr = d[j, 0]
    r = np.floor(s * N)
    R = np.array([[0, r], [r, N]]).astype(int)
    X1 = Xr[R[0, 0]:R[0, 1], :]
    d1 = dr[R[0, 0]:R[0, 1]]
    X2 = Xr[R[1, 0]:R[1, 1], :]
    d2 = dr[R[1, 0]:R[1, 1]]

    return X1, d1, X2, d2