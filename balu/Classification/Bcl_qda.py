# -*- coding: utf-8 -*-
import numpy as np
from .Bcl_construct import Bcl_construct
from .Bcl_outscore import Bcl_outscore


def Bcl_qda(*args):
    """ ds, options = Bcl_qda(X, d, Xt, op)  Training & Testing together
     options = Bcl_qda(X, d, op)     Training only
     ds, options = Bcl_qda(Xt, options) Testing only

     Toolbox: Balu
        QDA (quadratic discriminant analysis) classifier.

        Design data:
           X is a matrix with features (columns)
           d is the ideal classification for X
           options.p is the prior probability, if p is not given,
           it will be estimated proportional to the number of samples of each
           class.

        Test data:
           Xt is a matrix with features (columns)

        Output:
           ds is the classification on test data
           options['dmin'] contains min(d).
           options['Ck'] is covariance matrix of each class.
           optionsmc contains the centroids of each class.
           options.string is a 8 character string that describes the performed
           classification (in this case 'qda     ').

        Reference:
           Hastie, T.; Tibshirani, R.; Friedman, J. (2001): The Elements of
           Statistical Learning, Springer (pages 84-90)

        Example: Training & Test together:
            from balu.ImagesAndData import balu_load
            from balu.Classification import Bcl_qda
            from balu.PerformanceEvaluation import Bev_performance

            data = balu_load('datagauss')           # simulated data (2 classes, 2 features)
            X = data['X']
            Xt = data['Xt']
            d = data['d']
            dt = data['dt']
            Bio_plotfeatures(X, d)                  # plot feature space
            op = {'p': []}
            ds, _ = Bcl_qda(X, d, Xt, op)           # QDA classifier
            p = Bev_performance(ds, dt)             # performance on test data
            print p

        Example: Training only
            from balu.ImagesAndData import balu_load
            from balu.Classification import Bcl_qda

            data = balu_load('datagauss')           # simulated data (2 classes, 2 features)
            X = data['X']
            Xt = data['Xt']
            d = data['d']
            dt = data['dt']
            Bio_plotfeatures(X, d)                  # plot feature space
            op = {'p': [0.75, 0.25]}                # prior probability for each class
            op = Bcl_qda(X, d, op)                  # QDA - training

        Example: Testing only (after training only example):
            ds, op = Bcl_qda(Xt, op)                # QDA - testing
            p = Bev_performance(ds, dt)             # performance on test data

        See also Blda.

     (c) GRIMA, PUC-DCC 2011 - D.Mery, E.Cortazar
     http://grima.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    train, test, X, d, Xt, options = Bcl_construct(args)

    if len(d.shape) < 2:
        d = np.expand_dims(d, axis=1)

    options = options.copy()

    options['string'] = 'qda     '
    if train:
        dmin = d.min()
        dmax = d.max()
        d = d - dmin + 1
        N = d.shape[0]               # number of samples
        K = int(dmax - dmin + 1)     # number of classes

        options['p'] = np.array(options['p'])
        pest = options['p'].size == 0

        if pest:
            p = np.zeros(K)
        else:
            p = options['p']

        m = X.shape[1]
        L = np.zeros((int(K), 1))

        mc = np.zeros((m, K))
        Ck = np.zeros((m, m, K))
        for k in range(int(K)):
            ii, _ = np.where(d == k + 1)            # index of rows of class k
            if ii.size == 0:
                print('Bcl_qda: There is no class {0} in the data.'.format(k + dmin))

            L[k] = ii.size                          # number of samples in class k
            Xk = X[ii, :]                           # samples of class k
            mc[:, k] = np.mean(Xk, axis=0)          # mean of class k
            Ck[:, :, k] = np.cov(Xk, rowvar=False)  # covariance of class k
            if pest:
                p[k] = L[k] / float(N)

        options['dmin'] = dmin
        options['mc'] = mc
        options['Ck'] = Ck
        options['p'] = p
        output = options

    if test:
        K = options['mc'].shape[1]
        Nt = Xt.shape[0]
        D = np.zeros((Nt, K))
        for k in range(K):
            C = options['Ck'][:, :, k]
            Xd = (Xt - np.dot(np.ones((Nt, 1)), options['mc'][:, k][:, np.newaxis].T))
            C1 = np.diag(-0.5 * np.dot(np.dot(Xd, np.linalg.pinv(C)), Xd.T))[:, np.newaxis]
            C2 = (-0.5 * np.log(np.linalg.det(C)) + np.log(options['p'][k])) * np.ones((Nt, 1))
            D[:, k] = np.squeeze(C1 + C2)

        i = np.max(D, axis=1)
        j = np.argmax(D, axis=1)
        sc = np.ones(i.shape) / (np.abs(i) + 1e-5)
        ds = j + options['dmin']
        ds = Bcl_outscore(ds, sc, options)
        output = ds, options

    return output
