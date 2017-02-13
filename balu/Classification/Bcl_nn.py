# -*- coding: utf-8 -*-
import numpy as np
from .Bcl_construct import Bcl_construct
from sklearn.neural_network import MLPClassifier


def Bcl_nn(*args):
    """ ds, options = Bcl_nn(X, d, Xt, op)  Training & Testing together
     options = Bcl_nn(X, d, op)     Training only
     ds, options = Bcl_nn(Xt, options) Testing only

     Toolbox: Balu
        Neural Network using scikit-learn's MLPClassifier.

        Design data:
           X is a matrix with features (columns)
           d is the ideal classification for X
           options['method'] = 0, 1, 2, 3 for 'identity','tanh' or 'relu' (default=3)
           options['iter'] is the max. number of iterations used in the MLPClassifier.fit algorithm
           (default=100).

        Test data:
           Xt is a matrix with features (columns)

        Output:
           ds is the classification on test data
           options['net'] contains information about the neural network
           (from class MLPClassifier in scikit-learn).
           options['dmin'] contains np.min(d).
           options['string'] is a 8 character string that describes the performed
           classification (e.g., 'nn,3 ' means relu - neural network).

        Example: Training & Test together:
            from balu.ImagesAndData import balu_load
            from balu.Classification import Bcl_nn
            from balu.PerformanceEvaluation import Bev_performance

            data = balu_load('datagauss')           # simulated data (2 classes, 2 features)
            X = data['X']
            Xt = data['Xt']
            d = data['d']
            dt = data['dt']
            Bio_plotfeatures(X, d)                  # plot feature space
            op = {'method': 3, 'iter': 12}
            ds, _ = Bcl_nn(X, d, Xt, op)            # logistic - neural network
            p = Bev_performance(ds, dt)             # performance on test data

        Example: Training only
            from balu.ImagesAndData import balu_load
            from balu.Classification import Bcl_nn

            data = balu_load('datagauss')           # simulated data (2 classes, 2 features)
            X = data['X']
            Xt = data['Xt']
            d = data['d']
            dt = data['dt']
            Bio_plotfeatures(X, d)                  # plot feature space
            op = {'method': 3, 'iter': 12}
            op = Bcl_nn(X, d, op)                   # logistic - neural network - training

        Example: Testing only (after training only example):
            ds, _ = Bcl_nn(Xt, op)                  # logistic - neural network - testing
            p = Bev_performance(ds, dt)             # performance on test data


        Implementation based on Bcl_nnlgm from Balu Matlab toolbox and Neural Network function
        from scikit-learn.

     D.Mery, PUC-DCC, 2010
     http://dmery.ing.puc.cl

     And

     D. Patiño (dapatinoco@unal.edu.co) (2016)
     https://github.com/dipaco
     """

    train, test, X, d, Xt, options = Bcl_construct(args)

    if len(d.shape) < 2:
        d = np.expand_dims(d, axis=1)

    options = options.copy()

    if 'iter' not in options:
        options['iter'] = 100

    options['string'] = 'nn,{0}    '.format(options['method'])
    if train:

        all_activation_functions = ['identity', 'logistic', 'tanh', 'relu']
        m = 'relu'
        c = options['method']
        if isinstance(c, int):
            if 0 <= c < len(all_activation_functions):
                m = all_activation_functions[c]
        elif c in all_activation_functions:
            m = c

        dmin = d.min()

        net = MLPClassifier(solver='lbfgs', activation=m, max_iter=options['iter'])
        net.fit(X, d.ravel())
        options['net'] = net
        options['dmin'] = dmin
        output = options

    if test:
        net = options['net']
        ds = net.predict(Xt)
        output = ds, options

    return output
