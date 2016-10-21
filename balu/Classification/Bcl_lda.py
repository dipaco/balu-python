# -*- coding: utf-8 -*-
import numpy as np
from .Bcl_construct import Bcl_construct
from .Bcl_outscore import Bcl_outscore


def Bcl_lda(*args):
    """ ds, options = Bcl_lda(X,d,Xt,[])  Training & Testing together
     options = Bcl_lda(X,d,[])     Training only
     ds, options = Bcl_lda(Xt,options) Testing only

     Toolbox: Balu
        LDA (linear discriminant analysis) classifier.
        We assume that the classes have a common covariance matrix

        Design data:
           X is a matrix with features (columns)
           d is the ideal classification for X
           options.p is the prior probability, if p is empty,
           it will be estimated proportional to the number of samples of each
           class.

        Test data:
           Xt is a matrix with features (columns)

        Output:
           ds is the classification on test data
           options['dmin'] contains np.min(d).
           options['Cw1'] is pinv(within-class covariance).
           options['mc'] contains the centroids of each class.
           options['string'] is a 8 character string that describes the performed
           classification (in this case 'lda     ').

        Reference:
           Hastie, T.; Tibshirani, R.; Friedman, J. (2001): The Elements of
           Statistical Learning, Springer (pages 84-90)

        Example: Training & Test together:

            from balu.ImagesAndData import balu_load
            from balu.Classification import Bcl_lda
            from balu.InputOutput import Bio_plotfeatures
            from balu.PerformanceEvaluation import Bev_performance

            data = balu_load('datagauss')           #simulated data (2 classes, 2 features)
            X = data['X']
            d = data['d']
            Xt = data['Xt']
            dt = data['dt']
            Bio_plotfeatures(X, d)                  # plot feature space
            op = {'p': []}
            ds, options = Bcl_lda(X, d, Xt, op)     # LDA classifier
            p = Bev_performance(ds, dt)             # performance on test data
            print p

        Example: Training only
            from balu.ImagesAndData import balu_load
            from balu.Classification import Bcl_lda
            from balu.InputOutput import Bio_plotfeatures

            data = balu_load('datagauss')           #simulated data (2 classes, 2 features)
            X = data['X']
            d = data['d']
            Xt = data['Xt']
            dt = data['dt']
            Bio_plotfeatures(X, d)                  # plot feature space
            op = {'p': [0.75, 0.25]}                # prior probability for each class
            op = Bcl_lda(X,d,op);                   # LDA - training

        Example: Testing only (after training only example):
            from balu.Classification import Bcl_lda

            ds, _ = Bcl_lda(Xt, op)                 # LDA - testing
            p = Bev_performance(ds, dt)             # performance on test data

        See also Bcl_qda.

     (c) GRIMA-DCCUC, 2011
     http://grima.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    train, test, X, d, Xt, options = Bcl_construct(args)
    options = options.copy()
    options['string'] = 'lda     '
    if len(d.shape) < 2:
        d = np.expand_dims(d, axis=1)

    if train:
        dmin = np.amin(d)
        dmax = np.amax(d)
        d = d - dmin + 1
        N = d.size              # number of samples
        K = int(dmax - dmin + 1)     # number of classes

        options['p'] = np.array(options['p'])
        pest = options['p'].size == 0

        p = np.zeros(K)
        if not pest:
            p[:] = options['p']

        m = X.shape[1]
        L = np.zeros((K, 1))
        Cw = np.zeros((m, m))

        mc = np.zeros((m, K))
        for k in range(int(K)):
            ii, _ = np.where(d == k + 1)           # index of rows of class k
            if ii.size == 0:
                print('Bcl_lda: There is no class {0} in the data.'.format(k + dmin))
                exit()
            L[k, 0] = ii.size                   # number of samples in class k
            Xk = X[ii, :]                       # samples of class k
            mc[:, k] = np.mean(Xk, axis=0).T   # mean of class k
            Ck = np.cov(Xk, rowvar=False)                     # covariance of class k
            Cw = Cw + Ck * (L[k, 0] - 1)        # within-class covariance

            if pest:
                p[k] = L[k, 0] / N

        Cw /= (N - K)
        options['Cw1'] = np.linalg.pinv(Cw)
        options['dmin'] = dmin
        options['mc'] = mc
        options['p'] = p
        output = options

    if test:
        K = options['mc'].shape[1]
        Nt = Xt.shape[0]
        D = np.zeros((Nt, K))
        for k in range(int(K)):
            C1 = np.dot(options['Cw1'], options['mc'][:, k][np.newaxis].T)
            C2 = (-0.5 * np.dot(options['mc'][:, k].T, C1) + np.log(options['p'][k])) * np.ones((Nt, 1))
            D[:, k] = np.squeeze(np.dot(Xt, C1) + C2)

        a = np.amax(D, axis=1)
        i, j = np.unravel_index(np.argmax(D, axis=1), D.shape)
        sc = np.ones(a.size) / (np.abs(a) + 1e-5)
        ds = j + options['dmin']
        ds = Bcl_outscore(ds, sc, options)
        output = ds, options

    return output
