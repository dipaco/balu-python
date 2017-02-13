import numpy as np
from balu.Classification import Bcl_structure
from .Bev_performance import Bev_performance
from scipy import stats


def Bev_crossval(X, d, options):
    """p, ci = Bev_crossval(X, d, options)

     Toolbox: Balu

        Cross-validation evaluation of a classifier.

        v-fold Cross Validation in v groups of samples X and classification d
        according to given method. If v is equal to the number of samples,
        i.e., v = size(X,1), this method works as the original
        cross-validation, where training will be in X without sample i and
        testing in sample i. ci is the confidence interval.

        X is a matrix with features (columns)
        d is the ideal classification for X

        options['b'] is a Balu classifier or several classifiers (see example)
        options['v'] is the number of groups (folders) of the cross-validations
        options['c'] is the probability of the confidence intervale.
        options['strat'] = True means the portions are stratified.
        options['show'] = True displays results.

        Example for one classifier without stratified data:
            from balu.ImagesAndData import balu_load
            from balu.InputOutput import Bio_plotfeatures
            from balu.PerformanceEvaluation import Bev_crossval

            data = balu_load('datagauss')                                         # simulated data (2 classes, 2 features)
            X = data['X']
            d = data['d']
            Bio_plotfeatures(X, d)                                                # plot feature space
            op = {
                'b': [
                    {'name': 'knn', 'options': {'k': 5}}                          # knn with 5 neighbors
                ],
                'strat': False,
                'v': 10,
                'c': 0.9,
                'show': False                                                     # 10 groups cross-validation for 90% confidence
            }

            p, ci = Bev_crossval(X, d, op)                                        # cross valitadion
            print(p)
            print(ci)

        Example for more classifiers with stratified data:
            from balu.ImagesAndData import balu_load
            from balu.InputOutput import Bio_plotfeatures
            from balu.PerformanceEvaluation import Bev_crossval

            data = balu_load('datagauss')                                         # simulated data (2 classes, 2 features)
            X = data['X']
            d = data['d']
            Bio_plotfeatures(X, d)                                                # plot feature space
            op = {
                'b': [
                    {'name': 'knn', 'options': {'k': 5}},                         # knn with 5 neighbors
                    {'name': 'knn', 'options': {'k': 7}},                         # % KNN with 7 neighbors
                    {'name': 'knn', 'options': {'k': 9}},                         # KNN with 9 neighbors
                    {'name': 'lda', 'options': {'p': []}},                        # LDA
                    {'name': 'qda', 'options': {'p': []}},                        # QDA
                    {'name': 'nn', 'options': {'method': 3, 'iter': 10}},         # Neural Network
                    {'name': 'svm', 'options': {'kernel': 2}},                    # rbf-svm
                    {'name': 'dmin', 'options': {}},                              # Euclidean distance
                    {'name': 'maha', 'options': {}}                               # Mahalanobis distance
                ],
                'strat': True,
                'v': 10,
                'c': 0.95,
                'show': True                                                      # 10 groups cross-validation for 90% confidence
            }

            p, ci = Bev_crossval(X, d, op)                                        # cross valitadion
            print(p)
            print(ci)                                     % cross valitadion

        The mean performance of classifier k is given in p(k), and the
        confidence intervals for c*100% are in ci(k,:).

     (c) Grima, PUC-DCC, 2011: D.Mery, E.Cortazar, B. Banerjee
     http://grima.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    v = options['v']
    b = options['b']
    c = options['c']

    if len(d.shape) < 2:
        d = np.expand_dims(d, axis=1)

    if 'strat' in options:
        strat = options['strat']
    else:
        strat = False

    if v == 1:
        print('Warning: cross validation with only one group means data training = data test.')

    if 'show' not in options:
        show = True
    else:
        show = options['show']

    n = len(b)
    N = X.shape[0]

    dmin = d.min()
    dmax = d.max()
    nn = range(dmin, dmax+1)

    p = np.zeros(n)
    ci = np.zeros((n, 2))
    for k in range(n):

        if v == 1:
            XX = X
            XXt = X
            dd = d
            ddt = d
        elif strat:
            temporal = [[None, None] for i in range(v)]
            for cl in range(dmin, dmax + 1):
                selec, _ = np.where(d == cl)
                XTemp = X[selec, :]
                dTemp = d[selec, :]

                numElem = selec.size

                # Checking if every class has at least v samples, otherwise raise Warning (Sandipan)
                if numElem < v:
                    print('Class {0} has only {1} samples, less than fold value {2}!!!\n', dTemp[0, 0], numElem, v)

                j = np.argsort(np.random.rand(numElem))
                r = int(np.floor(numElem/float(v)))

                XTemp = XTemp[j, :]
                dTemp = dTemp[j, :]

                for iTemp in range(v):
                    rango = np.arange(iTemp*r, (iTemp + 1)*r)

                    temporal[iTemp][0] = XTemp[rango, :] if temporal[iTemp][0] is None else np.vstack((temporal[iTemp][0], XTemp[rango, :]))
                    temporal[iTemp][1] = dTemp[rango, :] if temporal[iTemp][1] is None else np.vstack((temporal[iTemp][1], dTemp[rango, :]))

            Xr = None
            dr = None
            R = np.zeros((v, 2))
            ant = 0
            for iTemp in range(v):

                Xr = temporal[iTemp][0] if Xr is None else np.vstack((Xr, temporal[iTemp][0]))
                dr = temporal[iTemp][1] if dr is None else np.vstack((dr, temporal[iTemp][1]))

                next = dr.shape[0]
                R[iTemp, :] = np.array([ant, next-1])
                ant = next

        else:
            rn = np.random.rand(N)
            j = np.argsort(rn)

            Xr = X[j, :]
            dr = d[j, :]

            r = np.floor(N/float(v))
            R = np.zeros((v, 2))
            ini = 0
            for i in range(v-1):
                R[i, :] = np.array([ini, ini+r-1])
                ini += r

            R[-1, :] = np.array([ini, N-1])

        pp = np.zeros(v)
        for i in range(v):
            if v > 1:
                XXt = Xr[R[i, 0]:R[i, 1] + 1, :]
                ddt = dr[R[i, 0]:R[i, 1] + 1, :]
                XX = None
                dd = None
                for j in range(v):
                    if j != i:

                        XX = Xr[R[j, 0]:R[j, 1] + 1, :] if XX is None else np.vstack((XX, Xr[R[j, 0]:R[j, 1] + 1, :]))
                        dd = dr[R[j, 0]:R[j, 1] + 1, :] if dd is None else np.vstack((dd, dr[R[j, 0]:R[j, 1] + 1, :]))

            dds, ops = Bcl_structure(XX, dd, XXt, b[k])
            pp[i] = Bev_performance(ddt, dds, nn)

        p[k] = np.mean(pp)
        s = ops[0]['options']['string']

        # Confidence Interval
        if v > 1:
            pm = p[k]
            mu = pm
            sigma = np.sqrt(pm * (1 - pm) / float(N))

            if v > 20:
                p1, p2 = stats.norm.interval(c, loc=mu, scale=sigma)
            else:
                #Not sure if v or v - 1
                p1, p2 = stats.t.interval(c, v-1, loc=mu, scale=sigma)

            ci[k, :] = np.array([p1, p2])

            if show:
                print('{0:3d}) {1}  {2:.2f}% in ({3:.2f}, {4:.2f}%) with CI={5:.2f}% \n'.format(k, s, p[k]*100, p1*100, p2*100, c*100))
        else:
            ci[k, :] = np.array([0.0, 0.0])

    return p, ci
