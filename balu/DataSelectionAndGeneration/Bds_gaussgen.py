# -*- coding: utf-8 -*-
import numpy as np


def Bds_gaussgen(m, s, n):
    """ X, d = Bds_gaussgen(m, s, n)

     Toolbox: Balu

        Gaussian Random Sample Generator.
        m matrix qxp. m(i,j) is mean of class i for feature j
        s matrix qxp. s(i,j) is std of class i for feature j
        n vector qx1. n(i) is number of samples of class i

        Example for two classes and two features:
            import numpy as np
            from balu.InputOutput import Bio_plotfeatures
            from balu.DataSelectionAndGeneration import Bds_gaussgen

            mu = np.array([[10, 1], [1, 10]])
            st = 4 * np.ones((2, 2))
            X, d = Bds_gaussgen(mu, st, 500 * np.ones((2, 1)))
            Bio_plotfeatures(X, d)

        Example for three classes and two features:
            import numpy as np
            from balu.InputOutput import Bio_plotfeatures
            from balu.DataSelectionAndGeneration import Bds_gaussgen

            mu = np.array([[2, 1], [1, 2], [2, 2]])
            st = np.ones((3, 2)) / 4.0
            X, d = Bds_gaussgen(mu, st, 500 * np.ones((3, 1)))
            Bio_plotfeatures(X, d)

     (c) GRIMA-DCCUC, 2011
     http://grima.ing.puc.cl

    With collaboration from:
    Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    m = np.array(m)
    s = np.array(s)
    n = np.array(n)

    if len(n.shape) > 1:
        n = np.squeeze(n)

    n = n.astype(int)

    q = n.size                 # number of classes
    p = m.shape[1]             # number of features
    N = np.sum(n)              # number of samples
    X = np.zeros((N, p))
    d = np.zeros((N, 1))
    t = 0

    for i in range(q):
        d[t:t + n[i]] = i + 1
        x = np.zeros((n[i], p))
        for j in range(p):
            x[:, j] = np.random.normal(m[i, j], s[i, j], n[i])

        X[t:t + n[i], :] = x
        t = t + n[i]

    return X, d
