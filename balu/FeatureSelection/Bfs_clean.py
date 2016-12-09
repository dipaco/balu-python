# -*- coding: utf-8 -*-
import numpy as np
import warnings


def Bfs_clean(X, show=False):
    """select = Bfs_clean(X,show)

     Toolbox: Balu
        Feature selection cleaning.

        It eliminates constant features and correlated features.

        Input: X is the feature matrix.
               show = 1 displays results (default show=0)
        Output: selec is the indices of the selected features

     Example:

        from balu.FeatureSelection import Bfs_clean
        from balu.ImagesAndData import balu_load

        k = balu_load('datareal')
        s = Bfs_clean(f, 1)             # index of selected features
        X = f[:, s]                     # selected features
        Xn = fn[s]                      # list of feature names
        print 'Original:', f.shape      # original set of features
        print 'Selected:', X.shape      # set of selecte features

     D.Mery, PUC-DCC, Jul. 2009
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """
    f = X

    nf = f.shape[1]
    p = np.arange(0, nf)
    ip = np.zeros((nf, 1))

    # eliminating correlated features
    warnings.filterwarnings('ignore')
    if f.shape[0] == 1:
        return p
    else:
        C = np.abs(np.corrcoef(f, rowvar=0))
    ii, jj = np.where(C > 0.99)

    if ii.size > 0:
        for i in range(ii.size):
            if np.abs(ii[i] - jj[i]) > 0:
                k = max(ii[i], jj[i])
                t, = np.where(p == k)
                n = p.size
                if t.size > 0:
                    if t[0] == 1:
                        p = p[1:]
                    else:
                        if t[0] == n - 1:
                            p = p[:n - 1]
                        else:
                            # p = p[[1:t - 1  t + 1:n]];
                            p = np.delete(p, t)

    ip[p] = 1

    # eliminating constant features
    s = np.std(f, axis=0)
    ii, = np.where(s < 1e-8)
    if ii.size > 0:
        ip[ii] = 0

    p, _ = np.where(ip)
    fc = f[:, p]
    nc = fc.shape[1]
    if show:
        print('Bfs_clean: number of features reduced from {0} to {1}.\n'.format(nf, nc))

    return p
