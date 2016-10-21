# -*- coding: utf-8 -*-
import numpy as np
from .Bcl_construct import Bcl_construct
from .Bcl_outscore import Bcl_outscore


def Bcl_dmin(*args):
    """ ds, options = Bcl_dmin(X, d, Xt, None)  Training & Testing together
     options = Bcl_dmin(X, d, None)     Training only
     ds      = Bcl_dmin(Xt, options) Testing only

     Toolbox: Balu
        Classifier using Euclidean minimal distance

        Design data:
           X is a matrix with features (columns)
           d is the ideal classification for X

        Test data:
           Xt is a matrix with features (columns)

        Output:
           ds is the classification on test data
           options['mc'] contains the centroids of each class.
           options[´dmin'] contains np.min(d).
           options['string'] is a 8 character string that describes the performed
           classification (in this case 'dmin    ').

        Example: Training & Test together:
            from balu.ImagesAndData import balu_load
            from balu.Classification import Bcl_dmin
            from balu.PerformanceEvaluation import Bev_performance

            data = balu_load('datagauss')           # simulated data (2 classes, 2 features)
            X = data['X']
            Xt = data['Xt']
            d = data['d']
            dt = data['dt']
            Bio_plotfeatures(X, d)                  # plot feature space
            ds, _ = Bcl_dmin(X, d, Xt, None)        # Euclidean distance classifier
            p = Bev_performance(ds, dt)             # performance on test data

        Example: Training only
            from balu.ImagesAndData import balu_load
            from balu.Classification import Bcl_dmin

            data = balu_load('datagauss')           # simulated data (2 classes, 2 features)
            X = data['X']
            Xt = data['Xt']
            d = data['d']
            dt = data['dt']
            Bio_plotfeatures(X, d)                  # plot feature space
            op = Bcl_dmin(X, d, None)               # Euclidean distance classifier

        Example: Testing only (after training only example):
            ds, _ = Bcl_dmin(Xt, op)                   # Euclidean distance classifier - testing
            p = Bev_performance(ds, dt)             # performance on test data

        See also Bcl_maha.

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

    options['string'] = 'dmin    '

    if train:
        m = X.shape[1]
        dmin = d.min()
        dmax = d.max()
        d = d - dmin + 1
        n = int(dmax - dmin + 1)
        mc = np.zeros((n, m))

        for i in range(int(n)):
            ii, _ = np.where(d == i + 1)
            mc[i, :] = np.mean(X[ii, :], axis=0)

        options['mc'] = mc
        options['dmin'] = dmin
        output = options

    if test:
        mc = options['mc']
        n = mc.shape[0]
        Nt = Xt.shape[0]
        ds = np.zeros((Nt, 1))
        sc = np.zeros((Nt, 1))

        for q in range(Nt):
            D = np.dot(np.ones((n, 1)), Xt[q, :][np.newaxis]) - mc
            e = np.sum(D * D, axis=1)
            i = np.min(e)
            j = np.argmin(e)
            ds[q, 0] = j
            sc[q, 0] = i

        ds = ds + options['dmin']
        ds = Bcl_outscore(ds, sc, options)
        output = ds, options

    return output
