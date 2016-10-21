# -*- coding: utf-8 -*-
import numpy as np


def Bcl_construct(args):
    """ This function is not a classifier!!!
     This function is called by Balu classifier functions (such as Bcl_lda) to
     build the training and testing data.
    """
    train = False
    test = False
    nargin = len(args)
    if nargin == 2:
        #case 2 -> ds, options = Bcl_svm(Xt, options) # testing only
        X = np.array([])
        d = np.array([])
        Xt = args[0]
        options = args[1]
        test = True
    elif nargin == 3:
        #case 3 -> options = Bcl_svm(X, d, options) # training only
        X = args[0]
        d = args[1].astype(float)
        if X.shape[0] != d.shape[0]:
            print('Length of label vector does not match number of instances.')
            exit()

        Xt = np.array([])
        options = args[2]
        train = True
    elif nargin == 4:
        #case 4 -> ds, options = Bcl_svm(X, d, Xt, options) # training & test
        X = args[0]
        d = args[1].astype(float)
        if X.shape[0] != d.shape[0]:
            print('Length of label vector does not match number of instances.')
            exit()

        Xt = args[2]
        options = args[3]
        test = True
        train = True
    else:
        print('Bcl_construct: number of input arguments must be 2, 3 or 4.')
        exit()

    return train, test, X, d, Xt, options
