# -*- coding: utf-8 -*-
import numpy as np
from matplotlib.pyplot import show, clf, gca, close, figure, title, scatter
from .Bio_plotfeatures import Bio_plotfeatures
from balu.Classification import Bcl_structure


def Bio_decisionline(X, d, Xn, op):
    """ Bio_decisionline(X, d, Xn, op)

     Toolbox: Balu

        Diaplay a 2D feature space and decision line.

        X: Sample data
        d: classification of samples
        op: output of a trained classifier.

        Example:
            from balu.ImagesAndData import balu_load
            from balu.Classification import Bcl_structure
            from balu.InputOutput import Bio_decisionline
            from matplotlib.pyplot import close

            data = balu_load('datagauss')                       # simulated data(2 classes, 2 features)
            X = data['X']
            d = data['d']
            Xn = ['\beta_1', '\beta_2']

            b = [
                {'name': 'knn', 'options': {'k': 5  }},         # KNN with 5 neighbors
                {'name': 'lda', 'options': {'p': [] }},         # LDA
                {'name': 'qda', 'options': {'p': [] }},         # QDA
                {'name': 'svm', 'options': {'kernel': 3}},      # rbf-SVM
            ]
            op = b
            struct = Bcl_structure(X, d, op)
            close('all')
            Bio_decisionline(X, d, Xn, struct)

     (c) D.Mery, PUC-DCC, 2011
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
     """

    if len(d.shape) < 2:
        d = np.expand_dims(d, axis=1)

    if X.shape[1] != 2:
        print('Bio_decisionline works for two features only.')
        exit()

    clf()
    Bio_plotfeatures(X, d, Xn, hold=False)
    n = len(op)

    ax = gca()
    xl = ax.get_xlim()
    #x = xl[0]:s:xl[1]
    #x = np.linspace(xl[0], xl[1], int((xl[1] - xl[0]) / s))
    x = np.linspace(xl[0], xl[1], 150)
    nx = x.size
    yl = ax.get_ylim()
    #y = yl[0]:s:yl[1]
    #y = np.linspace(yl[0], yl[1], int((yl[1] - yl[0]) / s))
    y = np.linspace(yl[0], yl[1], 150)
    ny = y.size
    rx = np.dot(np.ones((ny, 1)), x[np.newaxis, :])
    ry = np.dot(y[:, np.newaxis], np.ones((1, nx)))
    XXt = np.vstack((rx.ravel(), ry.ravel())).T
    op = Bcl_structure(X, d, op)
    dt, _ = Bcl_structure(XXt, op)

    dmin = d.min().astype(int)
    dmax = d.max().astype(int)
    scol = 'ycwkbr'
    tcol = 'brgywk'

    close('all')
    for k in range(n):
        figure()
        Bio_plotfeatures(X, d, Xn, hold=False)
        title(op[k]['options']['string'])

        for i in range(dmin, dmax + 1):
            ii = np.where(dt[:, k] == i)[0]
            scatter(XXt[ii, 0], XXt[ii, 1], color=scol[i - dmin], marker='o')

        for i in range(dmin, dmax + 1):
            ii, jj = np.where(d == i)
            scatter(X[ii, 0], X[ii, 1], color=tcol[i - dmin], marker='x')

    show()
