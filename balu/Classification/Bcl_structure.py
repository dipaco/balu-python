# -*- coding: utf-8 -*-
import numpy as np
from Bcl_construct import Bcl_construct
from Bcl_outscore import Bcl_outscore


def Bcl_structure(*args):
    """ ds      = Bcl_structure(X,d,Xt,options)  Training & Testing together
     options = Bcl_structure(X,d,options)     Training only
     ds      = Bcl_structure(Xt,options)      Testing only

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
           load datagauss                                                        % simulated data (2 classes, 2 features)
           b(1).name = 'knn';   b(1).options.k = 5;                              % KNN with 5 neighbors
           b(2).name = 'knn';   b(2).options.k = 7;                              % KNN with 7 neighbors
           b(3).name = 'knn';   b(3).options.k = 9;                              % KNN with 9 neighbors
           b(4).name = 'lda';   b(4).options.p = [];                             % LDA
           b(5).name = 'qda';   b(5).options.p = [];                             % QDA
           b(6).name = 'nnglm'; b(6).options.method = 3; b(6).options.iter = 10; % Nueral network
           b(7).name = 'svm';   b(7).options.kernel = 4;                         % rbf-SVM
           b(8).name = 'maha';  b(8).options = [];                               % Euclidean distance
           b(9).name = 'dmin';  b(9).options = [];                               % Mahalanobis distance
           op = b;
           ds = Bcl_structure(X,d,Xt,op);                                        % ds has 9 columns
           p = Bev_performance(ds,dt)                                            % p has 9 performances


        Example: Training only
           load datagauss                                                        % simulated data (2 classes, 2 features)
           b(1).name = 'knn';   b(1).options.k = 5;                              % KNN with 5 neighbors
           b(2).name = 'knn';   b(2).options.k = 7;                              % KNN with 7 neighbors
           b(3).name = 'knn';   b(3).options.k = 9;                              % KNN with 9 neighbors
           b(4).name = 'lda';   b(4).options.p = [];                             % LDA
           b(5).name = 'qda';   b(5).options.p = [];                             % QDA
           b(6).name = 'nnglm'; b(6).options.method = 3; b(6).options.iter = 10; % Nueral network
           b(7).name = 'svm';   b(7).options.kernel = 4;                         % rbf-SVM
           b(8).name = 'maha';  b(8).options = [];                               % Euclidean distance
           b(9).name = 'dmin';  b(9).options = [];                               % Mahalanobis distance
           op = b;
           op = Bcl_structure(X,d,op);                                           % Training only

        Example: Testing only (after training only example):
           ds = Bcl_structure(Xt,op);                                            % Testing only
           p  = Bev_performance(ds,dt)

        See also Bcl_exe.

     D.Mery, PUC-DCC, Jul 2009
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """
    train, test, X, d, Xt, options = Bcl_construct(args)
    options = options.copy()

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
        ds = options

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

    return ds, options
