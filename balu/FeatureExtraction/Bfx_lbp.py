# -*- coding: utf-8 -*-
import numpy as np
from skimage.feature import local_binary_pattern
from balu.ImageProcessing import Bim_inthist
from scipy.signal import convolve2d, order_filter
from skimage.util.shape import view_as_blocks
from skimage.filters import median
from skimage.morphology import square


def Bfx_lbp(I, R=None, options={}):
    """
     X, Xn, options = Bfx_lbp(I, R, options)

     Toolbox: Balu
        Local Binary Patterns features

        X is the features vector, Xn is the list of feature names (see Example
        to see how it works).

        It calculates the LBP over the a regular grid of patches. The function
        uses scikit-image's local_binary_pattern implementation
        (see http://scikit-image.org/docs/dev/api/skimage.feature.html#local-binary-pattern).

        It returns a matrix of uniform lbp82 descriptors for I, made by
        concatenating histograms of each grid cell in the image.
        Grid size is options['hdiv'] * options['vdiv']

        R is a binary image or empty. If R is given the lbp will be computed
        the corresponding pixles R==0 in image I will be set to 0.

        options['mappingtype'] can have one of this options: {'nri_uniform', 'uniform', 'ror', 'default'}.
        If not options is provided 'nri_uniform' which produces histograms of 59 bins will be used.

         Output:
         X is a matrix of size ((hdiv*vdiv) x 59), each row has a
             histogram corresponding to a grid cell. We use 59 bins.
         options['x'] of size hdiv*vdiv is the x coordinates of center of ith grid cell
         options['y'] of size hdiv*vdiv is the y coordinates of center of ith grid cell
         Both coordinates are calculated as if image was a square of side length 1.

         References:
         Ojala, T.; Pietikainen, M. & Maenpaa, T. Multiresolution gray-scale
         and rotation invariant texture classification with local binary
         patterns. IEEE Transactions on Pattern Analysis and Machine
         Intelligence, 2002, 24, 971-987.

         Mu, Y. et al (2008): Discriminative Local Binary Patterns for Human
         Detection in Personal Album. CVPR-2008.

         Example 1:
            import numpy as np
            from balu.ImagesAndData import balu_imageload
            from balu.FeatureExtraction import Bfx_lbp
            from balu.InputOutput import Bio_printfeatures
            from matplotlib.pyplot import bar, figure, show

            options = {
                'weight': 0,                    # Weigth of the histogram bins
                'vdiv': 3,                      # one vertical divition
                'hdiv':  3,                     # one horizontal divition
                'samples': 8,                   # number of neighbor samples
                'mappingtype': 'nri_uniform'    # uniform LBP
            }
            I = balu_imageload('testimg1.jpg')  # input image
            J = I[119:219, 119:239, 1]          # region of interest (green)
            figure(1)
            imshow(J, cmap='gray')              # image to be analyzed
            X, Xn = Bfx_lbp(J, None, options)   # LBP features
            figure(2);
            bar(np.arange(X.shape[1]), X[0, :])
            Bio_printfeatures(X, Xn)
            show()

         Example 2:
            import numpy as np
            from balu.ImagesAndData import balu_imageload
            from balu.FeatureExtraction import Bfx_lbp
            from balu.InputOutput import Bio_printfeatures
            from matplotlib.pyplot import bar, figure, show

            options = {
                'weight': 0,                    # Weigth of the histogram bins
                'vdiv': 3,                      # one vertical divition
                'hdiv':  3,                     # one horizontal divition
                'samples': 8,                   # number of neighbor samples
                'mappingtype': 'uniform'        # uniform LBP
            }
            I = balu_imageload('testimg1.jpg')  # input image
            J = I[119:219, 119:239, 1]          # region of interest (green)
            figure(1)
            imshow(J, cmap='gray')              # image to be analyzed
            X, Xn = Bfx_lbp(J, None, options)   # LBP features
            figure(2);
            bar(np.arange(X.shape[1]), X[0, :])
            Bio_printfeatures(X, Xn)
            show()

       See also Bfx_gabor, Bfx_clp, Bfx_fourier, Bfx_dct.

     (c) GRIMA-DCCUC, 2011
     http://grima.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2017)
    """

    if R is None:
        R = np.ones(I.shape)

    if 'show' not in options:
        options['show'] = False

    if 'normalize' not in options:
        options['normalize'] = False

    if options['show']:
        print('--- extracting local binary patterns features...')

    if 'samples' not in options:
        options['samples'] = 8

    if 'integral' not in options:
        options['integral'] = False

    if 'radius' not in options:
        options['radius'] = np.log(options['samples']) / np.log(2.0) - 1

    if 'weight' not in options:
        options['weight'] = 0

    LBPst = 'LBP'

    if 'mappingtype' not in options:
        options['mappingtype'] = 'nri_uniform'

    if options['mappingtype'] == 'ror':
        num_patterns = 256
    elif options['mappingtype'] == 'uniform':
        num_patterns = 10
    elif options['mappingtype'] == 'nri_uniform':
        num_patterns = 59
    else:
        options['mappingtype'] = 'default'
        num_patterns = 256

    st = '{0},{1}'.format(options['samples'], options['mappingtype'])

    # Get lbp image
    if R is not None:
        I[np.where(R == 0)] = 0

    radius = options['radius']
    P = options['samples']
    LBP = local_binary_pattern(I, P=P, R=radius, method=options['mappingtype'])
    n1, n2 = LBP.shape
    options['Ilbp'] = LBP

    if options['integral']:
        options['Hx'] = Bim_inthist(LBP, num_patterns)

    vdiv = options['vdiv']
    hdiv = options['hdiv']

    modn1 = n1 % vdiv
    if modn1 != 0:
        LBP = np.concatenate((LBP.T, np.zeros((LBP.shape[1], vdiv - modn1))), 1).T
        I = np.concatenate((I.T, np.zeros((I.shape[1], vdiv - modn1))), 1).T

    modn2 = n2 % hdiv
    if modn2 != 0:
        LBP = np.concatenate((LBP, np.zeros((LBP.shape[0], hdiv - modn2))), 1)
        I = np.concatenate((I, np.zeros((I.shape[0], hdiv - modn2))), 1)

    n1, n2 = LBP.shape

    ylen = int(np.round(n1/vdiv))
    xlen = int(np.round(n2/hdiv))
    # split image into blocks (saved as columns)
    grid_img = view_as_blocks(LBP, block_shape=(ylen, xlen))
    if options['weight'] > 0:
        LBPst = 'w' + LBPst
        mt = int(2 * radius - 1)
        mt2 = float(mt**2)
        Id = I.astype(int)

        weight = options['weight']
        if weight == 1:
            W = np.abs(convolve2d(Id, np.ones((mt, mt)) / mt2, mode='same') - Id)
        elif weight == 2:
            W = (np.abs(convolve2d(Id, np.ones((mt, mt)) / mt2, mode='same') - Id)) / (Id + 1)
        elif weight == 3:
            W = np.abs(median(Id, square(mt)) - Id)
        elif weight == 4:
            W = np.abs(median(Id, square(mt)) - Id) / (Id + 1)
        elif weight == 5:
            W = np.abs(order_filter(Id, np.ones((mt, mt)), 0) - Id)
        elif weight == 6:
            W = np.abs(order_filter(Id, np.ones((mt, mt)), 0) - Id) / (Id + 1)
        elif weight == 7:
            Id = convolve2d(Id, np.ones((mt, mt)) / mt2, mode='same')
            W = np.abs(order_filter(Id, np.ones((mt, mt)), 0) - Id) / (Id + 1)
        elif weight == 8:
            Id = median(Id, square(mt))
            W = np.abs(order_filter(Id, np.ones((mt, mt)), 0) - Id) / (Id + 1)
        elif weight == 9:
            Id = median(Id, square(mt))
            W = np.abs(order_filter(Id, np.ones((mt, mt)), 1) - Id) / (Id + 1)
        else:
            print("Bfx_lbp does not recognize options['weight'] = {0}.".format(options['weight']))

        grid_W = view_as_blocks(W, block_shape=(ylen, xlen))
        num_rows_blocks, num_cols_blocks = grid_W.shape[0:2]
        desc = np.zeros((num_patterns, num_cols_blocks*num_rows_blocks))
        p = 0
        for br in range(num_rows_blocks):
            for bc in range(num_cols_blocks):
                x = grid_img[br, bc].astype(int).ravel()
                y = grid_W[br, bc].ravel()
                d = np.zeros(num_patterns)
                for k in range(ylen * xlen):
                    d[x[k]] += y[k]
                desc[:, p] = d
                p += 1
    else:
        desc = (np.histogram(grid_img.ravel(), num_patterns)[0])[None]
        # calculate coordinates of descriptors as if it was square w/ side=1

    dx = 1.0 / float(hdiv)
    dy = 1.0 / float(vdiv)
    x = np.linspace(dx / 2.0, 1 - dx / 2.0, hdiv)
    y = np.linspace(dy / 2.0, 1 - dy / 2.0, vdiv)
    options['x'] = x
    options['y'] = y

    D = desc.T

    M, N = D.shape
    Xn = (N*M)*[None]
    X = np.zeros((1, N*M))
    k = 0
    for i in range(M):
        for j in range(N):
            Xn[k] = '{0}({1},{2})[{3}]                       '.format(LBPst, i, j, st)
            X[0, k] = D[i, j]
            k += 1

    if options['normalize']:
        X = X / np.sum(X)

    return X, Xn
