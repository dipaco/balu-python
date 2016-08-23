# -*- coding: utf-8 -*-
import numpy as np
import warnings
from matplotlib.pyplot import xlabel, ylabel, plot, show, legend, title, text, scatter

def Bio_plotfeatures(X, d, Xn=[]):
    """Bio_plotfeatures(X, d, Xn)

     Toolbox: Balu

     Plot features X according classification d. If the feature names are
     given in Xn then they will labeled in each axis.

     For only one feature, histograms are ploted.
     For two( or three) features, plots in 2D( or 3D) are given.
     For m > 3 features, m x m 2D plots are given(feature i vs.feature j)

     Example 1: 1D & 2D
     load datagauss                         # simulated data(2 classes, 2 features)
     figure(1)
     Bio_plotfeatures(X(:, 1), d, 'x1')     # histogram of feature 1
     figure(2)
     Bio_plotfeatures(X, d)                 # plot feature space in 2D(2 features)

     Example 2: 3D
     load datareal                          # real data
     X = f(:, [221 175 235]);               # only three features are choosen
     Bio_plotfeatures(X, d)                 # plot feature space in 3D(3 features)

     Example 3: 5D(using feature selection)
     load datareal                          # real data
     op.m = 5;                              # 5 features will be selected
     op.s = 0.75;                           # only 75 % of sample will be used
     op.show = 0;                           # display results
     op.b.name = 'fisher';                  # definition SFS with Fisher
     s = Bfs_balu(f, d, op);                # feature selection
     Bio_plotfeatures(f(:, s), d)           # plot feature space for 5 features

     See also Bev_roc.

     (c)
     GRIMA - DCCUC, 2011
     http: // grima.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    if len(X.shape) > 1:
        m = X.shape[1]
    else:
        m = 1

    if m > 9:
        print 'Bio_plotfeatures for {0} features makes {1} plots.'.format(m, m * m)
        exit()

    scflag = False
    if d.shape[1] == 2:
        sc = d[:, 1]
        d = d[:, 0]
        scflag = True

    dmin = np.min(d)
    dmax = np.max(d)

    col = 'gbrcmykbgrcmykbgrcmykbgrcmykgbrcmykbgrcmykbgrcmykbgrcmykgbrcmykbgrcmykbgrcmykbgrcmyk'
    mar = 'ox+v^ox+v^ox+v^ox+v^ox+v^ox+v^ox+v^ox+v^ox+v^ox+v^ox+v^ox+v^ox+v^ox+v^ox+v^ox+v^ox+v^ox+v^'

    # clf
    warnings.filterwarnings('ignore')
    s = []
    for k in range(dmin, dmax + 1):
        s.append('class {0}'.format(k - 1))

    if m < 4:
        if m == 1:
            for k in range(dmin, dmax + 1):
                idx, _ = np.where(d == k)
                print len(idx)
                h, x = np.histogram(X[idx], 100)
                dx = x[1] - x[0]
                A = np.sum(h) * dx
                x = np.hstack((x - dx / 2.0, x[-1] + dx))
                h = np.hstack((0, h, 0))
                plot(x, h / A, color=col[k])

            if len(Xn) == 0:
                xlabel(Xn)
            else:
                xlabel('feature value')

            ylabel('PDF')
        elif m == 2:
            if len(Xn) == 0:
                Xn = ['feature value 1', 'feature value 2']

            for k in range(dmin, dmax + 1):
                ii, _ = np.where(d == k)
                scatter(X[ii, 0], X[ii, 1], c=col[k], marker=mar[k])
                if scflag:
                    for ic in range(1, ii.size):
                        text(X[ii[ic], 0], X[ii[ic], 1], ['  ' + str(sc[ii[ic]])])

            title('feature space')
            xlabel(Xn[0])
            ylabel(Xn[1])
        elif m == 3:
            a = 5
        legend(s)
    else:
        a = 5

    show()
