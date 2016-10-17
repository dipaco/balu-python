# -*- coding: utf-8 -*-
import numpy as np
from warnings import filterwarnings
from skimage.morphology import dilation, square
from matplotlib.pyplot import imshow, show


def Bio_edgeview(B, E, cc=None, g=1, show_now=True):
    """ Bio_edgeview(B, E, c, g)

     Toolbox: Balu
       Display gray or color image I overimposed by color pixels determined
       by binary image E. Useful to display the edges of an image.
       Variable c is the color vector [r g b] indicating the color to be displayed
      (default: c = [1 0 0], i.e., red)
       Variable g is the number of pixels of the edge lines, default g = 1

     Example to display a red edge of a food:
        from balu.ImagesAndData import balu_imageload
        from balu.ImageProcessing import Bim_segbalu
        from balu.InputOutput import Bio_edgeview

        I = balu_imageload('testimg2.jpg')             # Input image
        R, E, J = Bim_segbalu(I)                       # Segmentation
        Bio_edgeview(I, E)

     D.Mery, PUC-DCC, Apr. 2008
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    %"""

    if cc is None:
        cc = np.array([1, 0, 0])

    B = B.astype(float)

    if B.max() > 1:
        B /= 256.0

    if len(B.shape) == 2:
        N, M = B.shape
        J = np.zeros((N, M, 3))
        J[:, :, 0] = B
        J[:, :, 1] = B
        J[:, :, 2] = B
        B = J

    B1 = B[:, :, 0]
    B2 = B[:, :, 1]
    B3 = B[:, :, 2]

    Z = B1 == 0
    Z = np.logical_and(Z, B2 == 0)
    Z = np.logical_and(Z, B3 == 0)
    ii, jj = np.where(Z == 1)
    if ii.size > 0:
        B1[ii, jj] = 1 / 256.0
        B2[ii, jj] = 1 / 256.0
        B3[ii, jj] = 1 / 256.0

    filterwarnings('ignore')
    E = dilation(E, square(g))
    ii, jj = np.where(E == 1)
    B1[ii, jj] = cc[0]
    B2[ii, jj] = cc[1]
    B3[ii, jj] = cc[2]
    Y = B.astype(float)
    Y[:, :, 0] = B1
    Y[:, :, 1] = B2
    Y[:, :, 2] = B3
    imshow((255*Y).astype('uint8'))
    if show_now:
        show()
