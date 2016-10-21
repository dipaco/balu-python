# -*- coding: utf-8 -*-
import numpy as np
import copy
from .Bcl_construct import Bcl_construct


def Bcl_structure(*args):
    """ ds = Bcl_structure(X, d, Xt, options)  Training & Testing together
     options = Bcl_structure(X, d, options)     Training only
     ds = Bcl_structure(Xt, options)      Testing only

     Toolbox: Balu
        Classification using Balu classifier(s) defined in structure b.

        Design data:
           X is a matrix with features (columns)
           d is the ideal classification for X
           options is a Balu classifier structure b with
              b.name      = Balu classifier's name
              b.options   = options of the classifier

           b can define one or more classifiers (see example).

        Test data:
           Xt is a matrix with features (columns)

        Output:
           ds is the classification on test data (one column per classifier)

        Example: Training & Test together:
            from balu.ImagesAndData import balu_load
            from balu.Classification import Bcl_structure
            from balu.PerformanceEvaluation import Bev_performance

            data = balu_load('datagauss')                       # simulated data(2 classes, 2 features)
            X = data['X']
            d = data['d']
            Xt = data['Xt']
            dt = data['dt']
            Xn = ['\beta_1', '\beta_2']

            b = [
                {'name': 'knn',  'options': {'k': 5  }},         # KNN with 5 neighbors
                {'name': 'knn',  'options': {'k': 7  }},         # KNN with 5 neighbors
                {'name': 'knn',  'options': {'k': 9  }},         # KNN with 5 neighbors
                {'name': 'lda',  'options': {'p': [] }},         # LDA
                {'name': 'qda',  'options': {'p': [] }},         # QDA
                {'name': 'nn' ,  'options': {'method': 2}},      # Neural Network
                {'name': 'svm',  'options': {'kernel': 3}},      # rbf-SVM
                {'name': 'maha', 'options': {}},                 # Mahalanobis distance
                {'name': 'dmin', 'options': {}},                 # Euclidean distance
            ]
            op = b
            ds, struct = Bcl_structure(X, d, Xt, op)             # ds has 9 columns
            p = Bev_performance(ds, dt)                          # p has 9 performances


        Example: Training only
            from balu.ImagesAndData import balu_load
            from balu.Classification import Bcl_structure

            data = balu_load('datagauss')                       # simulated data(2 classes, 2 features)
            X = data['X']
            d = data['d']
            Xt = data['Xt']
            dt = data['dt']
            Xn = ['\beta_1', '\beta_2']

            b = [
                {'name': 'knn',  'options': {'k': 5  }},         # KNN with 5 neighbors
                {'name': 'knn',  'options': {'k': 7  }},         # KNN with 5 neighbors
                {'name': 'knn',  'options': {'k': 9  }},         # KNN with 5 neighbors
                {'name': 'lda',  'options': {'p': [] }},         # LDA
                {'name': 'qda',  'options': {'p': [] }},         # QDA
                {'name': 'nn' ,  'options': {'method': 2}},      # Neural Network
                {'name': 'svm',  'options': {'kernel': 3}},      # rbf-SVM
                {'name': 'maha', 'options': {}},                 # Mahalanobis distance
                {'name': 'dmin', 'options': {}},                 # Euclidean distance
            ]
            op = b
            struct = Bcl_structure(X, d, op)                     # Training only

        Example: Testing only (after training only example):
            ds, _ = Bcl_structure(Xt, struct)                    # Testing only
            p = Bev_performance(ds, dt)

        See also Bcl_exe.

     D.Mery, PUC-DCC, Jul 2009
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """
    train, test, X, d, Xt, options = Bcl_construct(args)
    options = copy.deepcopy(options)

    b = options
    if not isinstance(b, list):
        b = [b]

    n = len(b)                      # number of classifiers

    if train:
        for i in range(n):
            Bn = b[i]['name']
            if Bn[0] != 'B':
                Bn = 'Bcl_' + Bn

            cl = getattr(__import__('balu').Classification, Bn)
            b[i]['options'] = cl(X, d, b[i]['options'])

        options = b
        output = options

    if test:
        nt = Xt.shape[0]
        ds3 = np.zeros((nt, n, 2))
        d3 = 0
        for i in range(n):
            Bn = b[i]['name']
            if Bn[1] != 'B':
                Bn = 'Bcl_' + Bn

            cl = getattr(__import__('balu').Classification, Bn)
            dsi, _ = cl(Xt, b[i]['options'])

            if len(dsi.shape) > 1:
                ds3[:, i, 0] = dsi[:, 0]
                if dsi.shape[1] == 2:
                    ds3[:, i, 1] = dsi[:, 1]
                    d3 = True
            else:
                ds3[:, i, 0] = dsi

        if d3:
            ds = ds3
        else:
            ds = ds3[:, :, 0]

        output = ds, options

    return output
