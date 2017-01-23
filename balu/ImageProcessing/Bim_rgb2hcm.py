# -*- coding: utf-8 -*-
import numpy as np
from scipy.misc import imresize
from scipy.optimize import minimize


def Bim_rgb2hcm(RGB):
    """ J = Bim_rgb2hcm(RGB)

     Toolbox: Balu
        Conversion RGB to high contrast image.

        RGB: color image
        J  : hcm image

      See details in:
      Mery, D.; Pedreschi, F. (2005): Segmentation of Colour Food Images using
      a Robust Algorithm. Journal of Food Engineering 66(3): 353-360.

      Example:
         I = imread('testimg2.jpg');
         J = Bim_rgb2hcm(I);
         figure(1)
         imshow(I); title('control image')
         figure(2)
         imshow(J); title('high contrast image')

     (c) D.Mery, PUC-DCC, 2011
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    RGB = RGB.astype(float)
    if len(RGB.shape) < 3:
        I = RGB
    else:
        RGB64 = imresize(RGB, (64, 64), interp='bicubic')
        #k = fminsearch(@Bstdmono,[1 1],[],RGB64)

        def f(k):
            return Bstdmono(k, RGB64)

        k = minimize(f, [1, 1])['x']
        I = k[0]*RGB[:, :, 0] + k[1]*RGB[:, :, 1] + RGB[:, :, 2]

    J = I - I.min()
    J = J / J.max()
    N, M = J.shape
    n = int(min(np.floor(J.shape[0] / 4.0), N))
    m = int(min(np.floor(J.shape[1] / 4.0), M))

    if np.mean(J[0:n, 0:m]) > 0.4:
        J = 1 - J

    return J

def Bstdmono(k, RGB):
    """ s = Bstdmono(k, RGB)

     Toolbox: Balu
        Standard deviation of normalized image I, where
        I = k(1)*R+k(2)*G+B (R = RGB(:,:,1), G=RGB(:,:,2), B = RGB(:,:,3)
        s: Standard deviation

        This function is called by Brgb2hcm, Bsegbalu.

     See also Brgb2hcm, Bsegbalu.

     D.Mery, PUC-DCC, Apr. 2008
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    I = k[0] * RGB[:, :, 0] + k[1] * RGB[:,:, 1] + RGB[:, :, 2]
    s = -np.std(I) / (I.max() - I.min())
    return s