# -*- coding: utf-8 -*-
import numpy as np
from .Bcl_construct import Bcl_construct
from .Bcl_outscore import Bcl_outscore
from sklearn.neighbors import KDTree
from scipy.stats import mode


def Bcl_knn(*args):
    """ ds, options = Bcl_knn(X, d, Xt, options)  Training & Testing together
     options = Bcl_knn(X, d, options)     Training only
     ds, _ = Bcl_knn(Xt, options)      Testing only

     Toolbox: Balu
        KNN (k-nearest neighbors) classifier using randomized kd-tree
        forest from FLANN. This implementation requires scikit-learn.

        Design data:
           X is a matrix with features (columns)
           d is the ideal classification for X
           options.k is the number of neighbors (default=10)

        Test data:
           Xt is a matrix with features (columns)

        Output:
           ds is the classification on test data
           options['kdtree'] contains information about the randomized kdtree
           (from KDTree function of scikit-learn).
           options['string'] is a 8 character string that describes the performed
           classification (e.g., 'knn,10  ' means knn with k=10).

        Example: Training & Test together:
            from balu.ImagesAndData import balu_load
            from balu.Classification import Bcl_knn
            from balu.PerformanceEvaluation import Bev_performance

            data = balu_load('datagauss')           # simulated data (2 classes, 2 features)
            X = data['X']
            Xt = data['Xt']
            d = data['d']
            dt = data['dt']
            Bio_plotfeatures(X, d)                  # plot feature space
            op = {'k': 10}
            ds, _ = Bcl_knn(X, d, Xt, op)           # knn with 10 neighbors
            p = Bev_performance(ds, dt)             # performance on test data
            print p

        Example: Training only
            from balu.ImagesAndData import balu_load
            from balu.Classification import Bcl_knn

            data = balu_load('datagauss')           # simulated data (2 classes, 2 features)
            X = data['X']
            Xt = data['Xt']
            d = data['d']
            dt = data['dt']
            Bio_plotfeatures(X, d)                  # plot feature space
            op = {'k': 10}
            op = Bcl_knn(X, d, op)                  # knn with 10 neighbors

        Example: Testing only (after training only example):
            ds, _ = Bcl_knn(Xt, op)                 # knn with 10 neighbors - testing
            p = Bev_performance(ds, dt)             # performance on test data


     D.Mery, C. Mena PUC-DCC, 2010-2013
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    train, test, X, d, Xt, options = Bcl_construct(args)
    options = options.copy()
    options['string'] = 'knn,{0:2d}  '.format(options['k'])

    if train:
        options['kdtree'] = KDTree(X, metric='euclidean')
        #options['X'] = X
        if len(d.shape) < 2:
            options['d'] = d[:, None]
        else:
            options['d'] = d
        output = options

    if test:
        kdtree = options['kdtree']
        dist, i = kdtree.query(Xt, k=options['k'], return_distance=True)

        if options['k'] > 1:
            ds = mode(options['d'][i.T][:, :, 0]).mode.T
        else: # for 1-KNN (modification by Carlos Mera)
            ds = options['d'][i.T][:, :, 0].T

        if 'output' in options:
            ns = ds.size
            sc = np.zeros((ns, 1))
            for q in range(ns):
                j, _ = np.where(options['d'][i.T[:, q]] == ds[q])
                sc[q] = np.min(dist.T[j, q]**2)

            ds = Bcl_outscore(ds, sc, options)

        output = ds, options

    return output