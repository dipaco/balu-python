# -*- coding: utf-8 -*-
import numpy as np
from .Bcl_construct import Bcl_construct
from sklearn import svm


def Bcl_svm(*args):
    """ ds      = Bcl_svm(X, d, Xt, options)  Training & Testing together
     options = Bcl_svm(X, d, options)     Training only
     ds      = Bcl_svm(Xt, options)      Testing only

     Toolbox: Balu
        Support Vector Machine approach using the scikit-learn library.

        Design data:
           X is a matrix with features (columns)
           d is the ideal classification for X

           options['kernel'] defines the SVM-kernel as follows:

           0: 'linear'      Linear kernel or dot product (default)
           1: 'poly'        Polynomial kernel (default order 3)
           2: 'rbf'         Gaussian Radial Basis Function kernel
           3: 'sigmoid'     Multilayer Perceptron kernel (default scale 1)

           kernel can be either int or string.

        Test data:
           Xt is a matrix with features (columns)

        Output:
           ds is the classification on test data
           options['svmStruct'] contains information about the trained classifier
           (from SVC class of scikit-learn).
           options.string is a 8 character string that describes the performed
           classification (e.g., 'svm,4  ' means rbf-SVM).

        Example: Training & Test together:
            from balu.ImagesAndData import balu_load
            from balu.Classification import Bcl_svm
            from balu.PerformanceEvaluation import Bev_performance

            data = balu_load('datagauss')           # simulated data (2 classes, 2 features)
            X = data['X']
            Xt = data['Xt']
            d = data['d']
            dt = data['dt']
            Bio_plotfeatures(X, d)                  # plot feature space
            op = {'kernel': 2}
            ds, _ = Bcl_svm(X, d, Xt, op)           # rbf-SVM classifier
            p = Bev_performance(ds, dt)             # performance on test data
            print p

        Example: Training only
            from balu.ImagesAndData import balu_load
            from balu.Classification import Bcl_svm
            from balu.PerformanceEvaluation import Bev_performance

            data = balu_load('datagauss')           # simulated data (2 classes, 2 features)
            X = data['X']
            Xt = data['Xt']
            d = data['d']
            dt = data['dt']
            Bio_plotfeatures(X, d)                  # plot feature space
            op = {'kernel': 2}
            op = Bcl_svm(X, d, op)                  # rbf-SVM classifier

        Example: Testing only (after training only example):
            ds, _ = Bcl_svm(Xt, op)                 # rbf-SVM classifier testing
            p = Bev_performance(ds, dt)             # performance on test data


     D.Mery, PUC-DCC, 2010
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
     """

    train, test, X, d, Xt, options = Bcl_construct(args)

    if len(d.shape) < 2:
        d = np.expand_dims(d, axis=1)

    options = options.copy()

    options['string'] = 'svm,{0}   '.format(options['kernel'])

    if train:

        all_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        k = 'rbf'
        c = options['kernel']
        if isinstance(c, int):
            if 0 <= c < len(all_kernels):
                k = all_kernels[c]
        elif c in all_kernels:
            k = c

        clf = svm.SVC()
        clf.fit(X, np.squeeze(d))
        options['svmStruct'] = clf
        output = options

    if test:
        clf = options['svmStruct']
        ds = clf.predict(Xt)[:, np.newaxis]
        output = ds, options

    return output
