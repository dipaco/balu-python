# -*- coding: utf-8 -*-
import numpy as np

def Bcl_lda(X):
    """ ds      = Bcl_lda(X,d,Xt,[])  Training & Testing together
     options = Bcl_lda(X,d,[])     Training only
     ds      = Bcl_lda(Xt,options) Testing only

     Toolbox: Balu
        LDA (linear discriminant analysis) classifier.
        We assume that the classes have a common covariance matrix

        Design data:
           X is a matrix with features (columns)
           d is the ideal classification for X
           options.p is the prior probability, if p is empty,
           it will be estimated proportional to the number of samples of each
           class.

        Test data:
           Xt is a matrix with features (columns)

        Output:
           ds is the classification on test data
           options.dmin contains min(d).
           options.Cw1 is inv(within-class covariance).
           options.mc contains the centroids of each class.
           options.string is a 8 character string that describes the performed
           classification (in this case 'lda     ').

        Reference:
           Hastie, T.; Tibshirani, R.; Friedman, J. (2001): The Elements of
           Statistical Learning, Springer (pages 84-90)

        Example: Training & Test together:
           load datagauss             % simulated data (2 classes, 2 features)
           Bio_plotfeatures(X,d)      % plot feature space
           op.p = [];
           ds = Bcl_lda(X,d,Xt,op);   % LDA classifier
           p = Bev_performance(ds,dt) % performance on test data

        Example: Training only
           load datagauss             % simulated data (2 classes, 2 features)
           Bio_plotfeatures(X,d)      % plot feature space
           op.p = [0.75 0.25];        % prior probability for each class
           op = Bcl_lda(X,d,op);      % LDA - training

        Example: Testing only (after training only example):
           ds = Bcl_lda(Xt,op);       % LDA - testing
           p = Bev_performance(ds,dt) % performance on test data

        See also Bcl_qda.

     (c) GRIMA-DCCUC, 2011
     http://grima.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """



    return 0