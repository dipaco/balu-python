# -*- coding: utf-8 -*-
import numpy as np
from .Bcl_construct import Bcl_construct
from .Bcl_outscore import Bcl_outscore


def Bcl_maha(*args):
    """ ds, options = Bcl_maha(X, d, Xt, None)  Training & Testing together
     options = Bcl_maha(X, d, None)     Training only
     ds, options = Bcl_maha(Xt, options) Testing only

     Toolbox: Balu
        Classifier using Mahalanobis minimal distance

        Design data:
           X is a matrix with features (columns)
           d is the ideal classification for X

        Test data:
           Xt is a matrix with features (columns)

        Output:
           ds is the classification on test data
           options['mc'] contains the centroids of each class.
           options['dmin'] contains min(d).
           options['Ck'] is covariance matrix of each class.
           options['string'] is a 8 character string that describes the performed
           classification (in this case 'maha    ').

        Example: Training & Test together:
            from balu.ImagesAndData import balu_load
            from balu.Classification import Bcl_maha
            from balu.PerformanceEvaluation import Bev_performance

            data = balu_load('datagauss')           # simulated data (2 classes, 2 features)
            X = data['X']
            Xt = data['Xt']
            d = data['d']
            dt = data['dt']
            Bio_plotfeatures(X, d)                  # plot feature space
            ds, _ = Bcl_maha(X, d, Xt, None)        # Euclidean distance classifier
            p = Bev_performance(ds, dt)             # performance on test data
            print p

        Example: Training only
            from balu.ImagesAndData import balu_load
            from balu.Classification import Bcl_maha
            from balu.PerformanceEvaluation import Bev_performance

            data = balu_load('datagauss')           # simulated data (2 classes, 2 features)
            X = data['X']
            Xt = data['Xt']
            d = data['d']
            dt = data['dt']
            Bio_plotfeatures(X, d)                  # plot feature space
            op = Bcl_maha(X, d, None)               # Euclidean distance classifier

        Example: Testing only (after training only example):
            ds, _ = Bcl_maha(op, Xt)                # Euclidean distance classifier - testing
            p = Bev_performance(ds, dt)             # performance on test data

        See also Xdmin.

     D.Mery, PUC-DCC, May 2010
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
     """

    train, test, X, d, Xt, options = Bcl_construct(args)

    if len(d.shape) < 2:
        d = np.expand_dims(d, axis=1)

    if options is None:
        options = {}

    options = options.copy()

    options['string'] = 'maha    '
    if train:
        m = X.shape[1]
        dmin = d.min()
        dmax = d.max()
        n = int(dmax - dmin + 1)
        d = d - dmin + 1
        mc = np.zeros((n, m))
        M = X.shape[1]
        Ck = np.zeros((M, M, n))
        for i in range(int(n)):
            ii, _ = np.where(d == i + 1)
            mc[i, :] = np.mean(X[ii, :], axis=0)
            CCk = np.cov(X[ii, :], rowvar=False)                     # covariance of class i
            Ck[:, :, i] = CCk

        options['mc'] = mc
        options['dmin'] = dmin
        options['Ck'] = Ck
        output = options

    if test:
        mc = options['mc']
        n = mc.shape[0]
        Nt = Xt.shape[0]
        ds = np.zeros((Nt, 1))
        sc = ds.copy()
        M = Xt.shape[1]

        Ck = np.zeros((M, M, n))
        for k in range(n):
            Ck[:, :, k] = np.linalg.pinv(options['Ck'][:, :, k])

        for q in range(Nt):
            dk = np.zeros((n, 1))
            for k in range(n):
                dx = Xt[q, :] - options['mc'][k, :]
                dk[k] = np.dot(dx, np.dot(Ck[:, :, k], dx.T))

            i = np.min(dk, axis=0)
            j = np.argmin(dk, axis=0)
            ds[q, 0] = j + 1
            sc[q, 0] = i

        ds = ds + options['dmin'] - 1
        ds = Bcl_outscore(ds, sc, options)
        output = ds, options

    return output
