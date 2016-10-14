# -*- coding: utf-8 -*-
import numpy as np
from skimage.morphology import closing, disk, remove_small_objects, remove_small_holes, erosion, square
from skimage.measure import perimeter


def Bim_morphoreg(J, t=None):
    """ R, E = Bmorphoreg(J, t)
     R, E = Bmorphoreg(Ro)

     Toolbox: Balu
        Morphology operations of binary image J>t (or Ro): remove isolate
        pixels and fill holes.
        R: binary image of the region
        E: binary image of the edge

      Example:
            from balu.ImagesAndData import balu_imageload
            from matplotlib.pyplot import figure, imshow, show
            from skimage.color import rgb2gray
            from balu.ImageProcessing import Bim_morphoreg, Bim_segotsu

            I = balu_imageload('testimg2.jpg')
            figure(1), imshow(I)
            J = rgb2gray(I.astype(float))
            Ro, E, J = Bim_segotsu(J)
            figure(2), imshow(Ro, cmap='gray')
            R, E = Bim_morphoreg(Ro)
            figure(3), imshow(R, cmap='gray')
            show()

     D.Mery, PUC-DCC, Apr. 2008
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    if t is None:
        Ro = J
    else:
        Ro = J > t

    A = remove_small_objects(Ro, int(np.floor(Ro.size / 100.0)), connectivity=2)
    C = closing(A, disk(7))
    R = remove_small_holes(C, np.floor(Ro.size / 100), connectivity=2)
    E = R - erosion(R, selem=square(3))

    return R, E