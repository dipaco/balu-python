# -*- coding: utf-8 -*-
import numpy as np


def Bfx_geo(R, options):
    """ X, Xn = Bfx_geo(R, options)
     X, Xn = Bfx_geo(L, options)

     Toolbox: Balu

        Gemteric feature extraction.

        This function calls gemetric feature extraction procedures of binary
        image R or labelled image L.

        X is the feature matrix (one feature per column, one sample per row),
        Xn is the list with the names of these features (see Example
        to see how it works).

       Example 1: Extraction of one region image
            from balu.FeatureExtraction import Bfx_geo
            from balu.ImageProcessing import Bim_segbalu
            from balu.InputOutput import Bio_printfeatures
            from balu.ImagesAndData import balu_imageload

            options = {'b': [
                {'name': 'hugeo',      'options': {'show': True}},                      # Hu moments
                {'name': 'flusser',    'options': {'show': True}},                      # Flusser moments
                {'name': 'fourierdes', 'options': {'show': True, 'Nfourierdes': 12}}    # Fourier descriptors
            ]}

            I = balu_imageload('testimg1.jpg')                      # input image
            R = Bim_segbalu(I)                                      # segmentation
            X, Xn = Bfx_geo(R, options)                             # geometric features
            Bio_printfeatures(X, Xn)

       Example 2: Extraction of multiple regions image
            from balu.FeatureExtraction import Bfx_geo
            from balu.ImageProcessing import Bim_segmowgli
            from balu.ImagesAndData import balu_imageload
            from matplotlib.pyplot import figure, imshow, title, hist, xlabel, show

            options = {'b': [
                {'name': 'hugeo',      'options': {'show': True}},                      # Hu moments
                {'name': 'basicgeo',   'options': {'show': True}},                      # basic geometric features
                {'name': 'fourierdes', 'options': {'show': True, 'Nfourierdes': 12}}    # Fourier descriptors
            ]}

            I = balu_imageload('rice.png')                          # input image
            R, m = Bim_segmowgli(I, np.ones(I.shape), 40, 1.5)      # segmentation
            X, Xn = Bfx_geo(R, options)                             # geometric features
            figure(); hist(X[:, 11]); xlabel(Xn[11])                # area histogram
            ii = np.where(np.abs(X[:, 19]) < 15)[0]                 # rice orientation
            K = np.zeros(R.shape)                                   # between -15 and 15 grad
            for i in range(ii.size):
                K = np.logical_or(K, R == ii[i])
            figure(); imshow(K, cmap='gray'); title('abs(orientation)<15 grad')
            show()

       See also Bfx_basicgeo, Bfx_hugeo, Bfx_flusser, Bfx_gupta,
                Bfx_fitellipse, Bfx_fourierdes, Bfx_files.

     (c) D.Mery, PUC-DCC, 2010
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """
    if R.size == 0:
        print('Bfx_geo: R is empty. Geometric features without segmentation has no sense.')
        exit()
    else:
        b = options['b']
        n = len(b)
        m = int(R.max())
        X = np.array([])
        for j in range(1, m + 1):
            Rj = R==j
            Xj = np.array([])
            Xnj = []
            for i in range(n):
                s = b[i]['name']
                if s[0:3] != 'Bfx':
                    s = 'Bfx_' + s

                balu_module = __import__('balu')
                if s not in dir(balu_module.FeatureExtraction):
                    print('Bfx_geo: function {0} does not exist.'.format(b[i]['name']))
                    exit()

                f = getattr(balu_module.FeatureExtraction, s)
                Xi, Xni = f(Rj, b[i]['options'])
                Xj = np.hstack((Xj, np.squeeze(Xi)))
                Xnj += Xni

            if X.size == 0:
                X = np.hstack((X, Xj))
            else:
                X = np.vstack((X, Xj))

        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        return X, Xnj