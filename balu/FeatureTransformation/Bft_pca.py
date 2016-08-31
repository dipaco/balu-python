# -*- coding: utf-8 -*-
import numpy as np


def Bft_pca(X, m):
    """
     Y,lambda,A,Xs,mx = Bft_pca(X,m)

     Toolbox: Balu
        Principal component analysis
        X is the matrix feature.
        m number of selected components or the energy 0<m<=1 (in this case it
        will be selected the first d principal components that fulfill the
        condition sum(lambda(1:d))/sum(lambda) >= energy. See Example 2)
        Y contains the m principal components of X
        lambda contains the standard deviation of each principal component
        A is the transformation matrix from X to Y
        Xs is the reconstructed X from A and Y.

      Example 1:
        import numpy as np
        from balu.ImagesAndData import balu_load, balu_imageload
        from matplotlib.pyplot import figure, imshow, show, bar
        from balu.FeatureTransformation import Bft_pca

        X = balu_imageload('cameraman.tif').astype(float) # 256x256 pixels
        Y, l, A, Xs, mm = Bft_pca(X, 30)    # 30 principal components
        figure(1)
        ind = np.arange(l.size)
        bar(ind, l/l[0])
        figure(2)
        imshow(np.hstack((X, Xs.astype(float))), cmap='gray')
        show()

      Example 2:
        import numpy as np
        from balu.ImagesAndData import balu_load, balu_imageload
        from matplotlib.pyplot import figure, imshow, show, bar
        from balu.FeatureTransformation import Bft_pca

        X = balu_imageload('cameraman.tif').astype(float) # 256x256 pixels
        Y, l, A, Xs, mx = Bft_pca(X, 0.999)   # 90% of energy
        figure(1)
        ind = np.arange(l.size)
        bar(ind, l/l[0])
        figure(2)
        imshow(np.hstack((X, Xs.astype(float))), cmap='gray')
        show()

     (c) GRIMA, PUC-DCC, 2011
     http://grima.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """
    #function[Y, lambda , A, Xs, mx, B] = Bft_pca(X, d, options)

    N, L = X.shape

    if m > 0 and m < 1:
        energy = m
        m = L
    else:
        energy = 0

    mx = np.mean(X, axis=0)
    mx = np.expand_dims(mx, 1).T
    MX = np.dot(np.ones((N, 1)), mx)
    X0 = X - MX
    Cx = np.cov(X.T)
    lbd, A = np.linalg.eig(Cx)

    #In python the eigenvalues are not necessarily ordered
    idx = lbd.argsort()[::-1]
    lbd = lbd[idx]
    A = A[:, idx]

    if energy > 0:
        sumlambda = np.sum(lbd)
        energylam = np.zeros((L, 1))
        for i in range(0, L):
            energylam[i, 0] = sum(lbd[0:i+1]) / sumlambda
        ii, _ = np.where(energylam > energy)
        m = ii[0]

    B = A[:, 0:m]
    Y = np.dot(X0, B)   # Y = X0 * A
    Y = Y[:, 0:m]   # the first m components
    Xs = np.dot(Y, B.T) + MX

    return Y, lbd, A, Xs, mx
