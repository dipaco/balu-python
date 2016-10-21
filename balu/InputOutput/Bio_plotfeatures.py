# -*- coding: utf-8 -*-
import numpy as np
import warnings
from matplotlib.pyplot import xlabel, ylabel, plot, show, legend, title, text, scatter, gcf, subplot, locator_params, clf
from mpl_toolkits.mplot3d import Axes3D


def Bio_plotfeatures(X, d, Xn=[], hold=True):
    """Bio_plotfeatures(X, d, Xn)

     Toolbox: Balu

     Plot features X according classification d. If the feature names are
     given in Xn then they will labeled in each axis.

     For only one feature, histograms are ploted.
     For two( or three) features, plots in 2D( or 3D) are given.
     For m > 3 features, m x m 2D plots are given(feature i vs.feature j)

     Example 1: 1D & 2D
        from balu.InputOutput import Bio_plotfeatures
        from balu.ImagesAndData import balu_load
        from matplotlib.pyplot import figure

        data = balu_load('datagauss')          # simulated data(2 classes, 2 features)
        X = data['X']
        d = data['d']
        figure(1)
        Bio_plotfeatures(X[:, 0], d, 'x1')     # histogram of feature 1
        figure(2)
        Bio_plotfeatures(X, d)                 # plot feature space in 2D(2 features)

     Example 2: 3D
        from balu.InputOutput import Bio_plotfeatures
        from balu.ImagesAndData import balu_load

        data = balu_load('datareal')          # real data
        f = data['f']
        d = data['d']
        X = f[:, [220, 174, 234]]             # only three features are choosen
        Bio_plotfeatures(X, d)                # plot feature space in 3D(3 features)

     Example 3: 5D(using feature selection)
         from balu.InputOutput import Bio_plotfeatures
         from balu.ImagesAndData import balu_load

         data = balu_load('datareal ')             # real data
         f = data['f']
         d = data['d']
         op['m'] = 5;                              # 5 features will be selected
         op['s'] = 0.75;                           # only 75 % of sample will be used
         op['show'] = False                        # display results
         op['b']['name'] = 'fisher';               # definition SFS with Fisher
         s = Bfs_balu(f, d, op);                   # feature selection
         Bio_plotfeatures(f[:, s], d)              # plot feature space for 5 features

     See also Bev_roc.

     (c)
     GRIMA - DCCUC, 2011
     http: // grima.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    clf()

    if len(X.shape) > 1:
        m = X.shape[1]
    else:
        m = 1

    if m > 9:
        print('Bio_plotfeatures for {0} features makes {1} plots.'.format(m, m * m))
        exit()

    scflag = False
    if len(d.shape) > 1:
        if d.shape[1] == 2:
            sc = d[:, 1]
            d = d[:, 0]
            scflag = True
        else:
            d = np.squeeze(d)

    dmin = int(np.min(d))
    dmax = int(np.max(d))

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
                idx = np.where(d == k)
                h, x = np.histogram(X[idx], 100)
                dx = x[1] - x[0]
                A = np.sum(h) * dx
                x = np.hstack((x - dx / 2.0, x[-1] + dx))
                h = np.hstack((0, h, 0))
                plot(x, h / A, color=col[k])

            if len(Xn) != 0:
                xlabel(Xn)
            else:
                xlabel('feature value')

            ylabel('PDF')
        elif m == 2:
            if len(Xn) == 0:
                Xn = ['feature value 1', 'feature value 2']

            for k in range(dmin, dmax + 1):
                ii = np.where(d == k)
                scatter(X[ii, 0], X[ii, 1], c=col[k], marker=mar[k])
                if scflag:
                    for ic in range(1, ii[0].size):
                        text(X[ii[0][ic], 0], X[ii[0][ic], 1], '  ' + str(sc[ii[0][ic]]))

            title('feature space')
            xlabel(Xn[0])
            ylabel(Xn[1])
        elif m == 3:

            fig = gcf()
            ax = fig.add_subplot(111, projection='3d')

            for k in range(dmin, dmax + 1):
                ii = np.where(d == k)
                ax.scatter(X[ii, 0], X[ii, 1], X[ii, 2], c=col[k], marker=mar[k])

                if scflag:
                    for ic in range(1, ii[0].size):
                        ax.text(X[ii[0][ic], 0], X[ii[0][ic], 1], X[ii[0][ic], 2], '  ' + str(sc[ii[0][ic]]))

            if len(Xn) > 0:
                ax.set_xlabel(Xn[0])
                ax.set_ylabel(Xn[1])
                ax.set_zlabel(Xn[2])
            else:
                ax.set_xlabel('feature value 1')
                ax.set_ylabel('feature value 2')
                ax.set_zlabel('feature value 3')
        legend(s)
    else:

        l = 1
        for j in range(1, m + 1):
            for i in range(1, m + 1):
                zi = X[:, i - 1]
                zj = X[:, j - 1]
                subplot(m, m, l)
                l += 1

                for k in range(dmin, dmax + 1):
                    ii = np.where(d == k)
                    scatter(zi[ii], zj[ii], c=col[k], marker=mar[k])
                    locator_params(nbins=3)

                    if scflag:
                        for ic in range(1, ii[0].size):
                            text(zi[ii[0][ic]], zj[ii[0][ic]], '  ' + str(sc[ii[0][ic]]))

                if len(Xn) > 0:
                    xl = Xn[i - 1]
                    yl = Xn[j - 1]
                else:
                    xl = 'z_{0}'.format(i - 1)
                    yl = 'z_{0}'.format(j - 1)

                if i == 1:
                    ylabel(yl)

                if j == m:
                    xlabel(xl)

    if hold:
        show()
