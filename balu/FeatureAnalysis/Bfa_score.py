# -*- coding: utf-8 -*-
import numpy as np
from .Bfa_mutualinfo import Bfa_mutualinfo
from .Bfa_relevance import Bfa_relevance
from .Bfa_mRMR import Bfa_mRMR
from .Bfa_jfisher import Bfa_jfisher
from .Bfa_sp100 import Bfa_sp100
from balu.PerformanceEvaluation import Bev_performance
from balu.Classification import Bcl_structure


def Bfa_score(X, d, options):
    """    J = Bfa_score(X, d, options)

         Toolbox: Balu
            Calculates a separability score for input data variables.
            X features matrix. X(i,j) is the feature j of sample i.
            d vector that indicates the ideal classification of the samples
            options score metric parameters

         See also Bfs_sfs.

         (c) D.Mery, PUC-DCC, 2011
         http://dmery.ing.puc.cl

         With collaboration from:
         Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
     """

    b = options['b']
    method = b['name'].lower()

    if method == 'mi':                                  # mutual information
        if 'param' not in b:
            dn = d.max() - d.min() + 1                  # number of classes
            p = np.ones((dn, 1)) / float(dn)
        else:
            p = b['param']

        Js = Bfa_mutualinfo(X, d, p)
    elif method == 'mr':                                # maximal relevance
        if 'param' not in b:
            dn = d.max() - d.min() + 1                  # number of classes
            p = np.ones((dn, 1)) / float(dn)
        else:
            p = b['param']

        Js = Bfa_relevance(X, d, p)
    elif method == 'mrmr':                              # minimal redundancy and maximal relevance
        if 'param' not in b:
            dn = d.max() - d.min() + 1                  # number of classes
            p = np.ones((dn, 1)) / float(dn)
        else:
            p = b['param']

        Js = Bfa_mRMR(X, d, p)
    elif method == 'fisher':                            # Fisher criterion
        if 'param' not in b:
            dn = d.max() - d.min() + 1  # number of classes
            p = np.ones((dn, 1)) / float(dn)
        else:
            p = b['param']

        Js = Bfa_jfisher(X, d, p)
    elif method == 'sp100':
        Js = Bfa_sp100(X, d)
    else:
        ds, _ = Bcl_structure(X, d, X, b)
        Js = Bev_performance(d, ds)

    return Js
