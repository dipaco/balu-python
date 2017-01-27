# -*- coding: utf-8 -*-
import numpy as np
from balu.InputOutput import Bio_statusbar
from balu.FeatureAnalysis import Bfa_score
import matplotlib.pyplot as plt


def Bfs_sfs(X, d, options={'show': True}, hold=False):
    """ selec = Bfs_sfs(X, d, options)

     Toolbox: Balu
        Sequential Forward Selection for fatures X according to ideal
        classification d. optins.m features will be selected.
        options['b']['method'] = 'fisher' uses Fisher objetctive function.
        options['b']['method'] = 'sp100' uses as criteria Sp @Sn=100%.
        options['b] can be any Balu classifier structure (see example).
        options['show'] = 1 display results.
        selec is the indices of the selected features.
        hold Keeps the bar graph opens until the user close it. Only
             works if show is true in the options

     Example 1: SFS using Fisher dicriminant
        data = balu_load('datareal')
        f = data['f']
        d = data['d']
        fn = data['fn']
        op = {
            'm': 10,                            # 10 features will be selected
            'show': True,                       # display results
            'b': {'name': 'fisher'}             # SFS with Fisher
        }
        s = Bfs_sfs(f, d, op)                   # index of selected features
        X = f[:, s]                             # selected features
        Xn = fn[s]                              # list of feature names
        op_lda = {'p': np.array([])}
        ds, op_lda = Bcl_lda(X, d, X, op_lda)   # LDA classifier
        p = Bev_performance(d, ds)              # performance with sfs
        print p

     Example 2: SFS using a KNN classifier
        from balu.ImagesAndData import balu_load
        from balu.FeatureSelection import Bfs_sfs

        data = balu_load('datareal')
        f = data['f']
        d = data['d']
        fn = data['fn']
        op = {
            'm': 10,                            # 10 features will be selected
            'show': True,                       # display results
            'b': {'name': 'knn',                # SFS with KNN with 5 neigbors
                  'options': {
                      'k': 5
                  }
            }
        }
        s = Bfs_sfs(f, d, op)                   # index of selected features
        X = f[:, s]                             # selected features
        Xn = fn[s]                              # list of feature names

     Example 3: SFS using sp100 criterion
        from balu.ImagesAndData import balu_load
        from balu.FeatureSelection import Bfs_sfs

        data = balu_load('datareal')
        f = data['f']
        d = data['d']
        fn = data['fn']
        op = {
            'm': 10,                            # 10 features will be selected
            'show': True,                       # display results
            'b': {'name': 'sp100'}              # SFS with sp100 criterion
        }
        s = Bfs_sfs(f, d, op)                   # index of selected features
        X = f[:, s]                             # selected features
        Xn = fn[s]                              # list of feature names

     (c) D.Mery, PUC-DCC, Jul. 2011
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """
    plt.ion()
    m = options['m']
    show = options['show']

    if 'force' not in options:
        options['force'] = False
    force = options['force']

    f = X;
    N = f.shape[1]
    selec = np.array([])              #selected features
    J = np.zeros((m, 1))
    # Jmax = 0
    k = 0

    if show:
        ff = Bio_statusbar('SFS progress')

    while k < m:

        fnew = False
        Jmax = float(np.finfo(np.float32).min)
        for i in range(N):
            if k == 0 or np.sum(selec == i) == 0:
                s = np.hstack((selec, i))
                fs = f[:, s.astype(int)]
                Js = Bfa_score(fs, d, options)
                if Js >= Jmax:
                    ks = i
                    Jmax = Js
                    fnew = True

        if fnew:
            selec = np.hstack((selec, ks))
            J[k] = Jmax
            k += 1

            if show:
                ff = Bio_statusbar(k / float(m), ff)
                plt.clf()
                plt.title('Selected features')
                plt.bar(np.arange(m) + 0.5, J)

                print(', Jmax = {0:8.4f}, Feature {1} selected.\n'.format(Jmax, int(selec[-1])))
                for i in range(selec.size):
                    plt.text(i+0.4, J[i]*1.05, '{0}'.format(int(selec[i])))

                plt.show(block=False)
                plt.pause(0.001)

        else:
            print('Bfs_sfs: no more improvement. Sequential search for feature selection is interrupted.')
            if force and k < m:
               print('Bfs_sfs: Warning! {0} random features were selected in order to have'.format(m-k))
               print('                  {0} selected features (options.force is true).\n'.format(m))
               t = np.arange(N)
               t = np.delete(t, selec)
               n = t.size
               x = np.random.rand(n)
               i = np.sort(x)
               selec = np.hstack((selec, t[i[0:m-k+1]]))

            k = float(np.finfo(np.float32).max)

    if show and hold:
        plt.pause(-1)
        #delete(ff)
    plt.ioff()

    return selec.astype(int)