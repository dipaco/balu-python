# -*- coding: utf-8 -*-
import numpy as np
from .Bev_confusion import Bev_confusion


def Bev_performance(d1, d2, nn=None):
    """p = Bev_erformance(d1,d2,nn)

     Toolbox: Balu
        Performance evaluation between two classifications, e.g., ideal (d1)
        and real (d2) classification.

        d1 and d2 are vectors or matrices (vector Nxn1 and Nxn2 respectivelly)
        at least n1 or n2 muts be one. N is the number of samples.
        p is the performance, diagonal sum divided by N.
        The classes should be labeled as 1, 2, ... n, but
        if nn = [i j] is given it indicates that the lowest
        class is min(nn) and the highest class is max(nn).
        Thus, n = max(nn)-min(nn)+1 and the size of T is nxn.

        d1 (or d2) can be a matrix (Nxn1 or Nxn2) containing the classification
        of n1 (or n2) classifiers. In this case the output p is a vector
        n1x1 (or n2x1) containing the performance of each classifier.
        If n1=1 and n2=1, then p is a scalar.

        Bperformance(d1,d2) and Bperformance(d2,d1) obtain the same result.

        Example:
           load datagauss             % simulated data (2 classes, 2 features)
           Bio_plotfeatures(X,d)      % plot feature space
           op.p = [];
           ds1 = Bcl_lda(X,d,Xt,op);  % LDA classifier
           ds2 = Bcl_qda(X,d,Xt,op);  % QDA classifier
           ds = [ds1 ds2];
           p = Bev_performance(ds,dt) % performance on test data

     D.Mery, PUC-DCC, 2010
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """
    d1 = d1.astype(float)
    d2 = d2.astype(float)

    if len(d1.shape) < 2:
        d1 = np.expand_dims(d1, axis=1)
    n1 = d1.shape[1]

    if len(d2.shape) < 2:
        d2 = np.expand_dims(d2, axis=1)
    n2 = d2.shape[1]

    if n1 == 1 or n2 == 1:

        if n2 > n1:
            ds = d2
            d = d1
        else:
            d = d2
            ds = d1

        n = ds.shape[1]
        p = np.zeros((n, 1))
        for i in range(n):
            if nn is not None:
                T, pp = Bev_confusion(d, ds[:, i], nn)
            else:
                T, pp = Bev_confusion(d, ds[:, i])
            p[i] = np.mean(pp)

    else:
        print('Bev_performance: at least d1 or d2 must have only one column')

    if p.size == 1:
        return p[0, 0]
    else:
        return p[:, 0]
