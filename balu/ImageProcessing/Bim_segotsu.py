# -*- coding: utf-8 -*-
import numpy as np
from skimage.filters import threshold_otsu
from .Bim_morphoreg import Bim_morphoreg
from .Bim_maxmin import Bim_maxmin
from skimage.color import rgb2gray


def Bim_segotsu(I, p=0):
    """ R = Bim_segotsu(I)

     Toolbox: Balu

        Otsu segmentation of a grayvalue. This function requires Image
        Processing Toolbox.

        Input data:
           I grayvalue image.

        Output:
           R: binary image.

        Example: Training & Test together:
            from balu.ImagesAndData import balu_imageload
            from matplotlib.pyplot import figure, imshow, show, title
            from balu.ImageProcessing import Bim_segotsu

            X = balu_imageload('testimg1.jpg')
            figure(1), imshow(X), title('original image')
            R, E, J = Bim_segotsu(X)
            figure(2)
            imshow(R, cmap='gray')
            title('segmented image')
            show()

      See also Bim_segbalu, Bim_segkmeans.

     (c) D.Mery, PUC-DCC, 2010
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    Id = I.astype(float)
    if len(I.shape) == 3 and I.shape[2] == 3:
        Id = rgb2gray(Id/256)

    J = Bim_maxmin(Id)
    n = int(J.shape[0] / 4.0)
    # Checks if the exterior border of the image is in fact part of the background
    # Otherwise it inverts the image
    if np.mean(J[0:n, 0:n]) > 0.4:
        J = 1 - J

    t = threshold_otsu(J)
    R, E = Bim_morphoreg(J, t+p)
    return R, E, J