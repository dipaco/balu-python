# -*- coding: utf-8 -*-
import numpy as np


def Bio_printfeatures(X, *args):
    """
     Bio_printfeatures(X, Xn)    # for features with feature names
     Bio_printfeatures(X)        # for feature values only

     Toolbox: Balu
        Display extracted features.
        Xn: feature names  (matrix mxp, one row per each string of p
            characters)
        X:  feature values (vector 1xm for m features)
        Xu: feature units  (matrix mxq, one row per each string of q
            characters)

        These variables are the outputs of Bgeofeatures or Bintfeatures.

        The output of Bio_printfeatures is like this:

        1 center of grav i       [pixels]         163.297106
        2 center of grav j       [pixels]         179.841850
        3 Height                 [pixels]         194.000000
        4 Width                  [pixels]         196.000000
        5 Area                   [pixels]         29361.375000
        :         :           :                  :

       Example 1: Display of standard geometric features of testimg1.jpg
          I = imread('testimg1.jpg');            # input image
          R = Bsegbalu(I);                       # segmentation
          [X,Xn,Xu] = Bfg_standard(R);           # standard geometric features
          Bprintfeatures(X,Xn,Xu)

       Example 2: Display of first 5 samples of datagauss.mat
          load datagauss
          Xn = ['[length]';'[weigh] '];
          Xu = ['cm';'kg'];
          for i=1:5
             fprintf('Sample %d:\n',i);
             Bprintfeatures(X(i,:),Xn,Xu)
             Benterpause
          end


       See also Bplotfeatures.

     (c) D.Mery, PUC-DCC, 2010
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    N = np.max(X.shape)
    if len(args) > 0:
        Xn = args[0]
    else:
        Xn = N * ['']

    for k in range(len(Xn)):
        print('{0} {1} {2}'.format(k, Xn[k], np.squeeze(X)[k]))