# -*- coding: utf-8 -*-
import scipy as sp
import numpy as np
from skimage.morphology import label, disk, dilation, remove_small_objects
from skimage.color import rgb2gray
from scipy.ndimage.morphology import binary_erosion, binary_closing
from skimage.filters import laplace, gaussian
import matplotlib.pyplot as plt


def Bim_segmowgli(J, R=None, Amin=20, sig=2):
    """  F, m = Bsegmowgli(J, R, Amin, sig)

     Toolbox: Balu
      Segmentation of regions in image J using LoG edge detection.
      R   : binary image of same size of J that indicates the piexels where
            the segmentation will be performed. Default R = ones(size(J));
      Amin: minimum area of the segmented details.
      sig : sigma of LoG edge detector.
      F   : labeled image of the segmentation.
      m   : numbers of segmented regions.

      Example 1:
            from balu.ImagesAndData import balu_imageload
            from balu.ImageProcessing import Bim_segmowgli
            from matplotlib.pyplot import figure, imshow, show, title

            I = balu_imageload('rice.png')
            figure(1), imshow(I, cmap='gray'), title('test image')
            F, m = Bim_segmowgli(I, R=None, Amin=40, sig=1.5)
            figure(2), imshow(F, cmap='gray'), title('segmented image')
            show()


      Example 2:
            from balu.ImagesAndData import balu_imageload
            from balu.ImageProcessing import Bim_segmowgli, Bim_segbalu
            from matplotlib.pyplot import figure, imshow, show, title

            I = balu_imageload('testimg4.jpg')
            figure(1), imshow(I), title('test image')
            R, E, J = Bim_segbalu(I, -0.1)
            figure(2), imshow(R, cmap='gray'), title('segmented object')
            G = I[:, :, 1]
            F, m = Bim_segmowgli(G, R=R, Amin=30, sig=2)
            figure(3), imshow(F), title('segmented regions')
            show()

         Another way to display the results:
            Bio_edgeview(I,F>0)

     See also Bsegbalu.

     D.Mery, PUC-DCC, Apr. 2008
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    if R is None:
        R = np.ones(J.shape, bool)

    if R.size == 0:
       R = np.ones(J.shape, bool)

    if len(J.shape) < 3:
        N, M = J.shape
        P = 1
    else:
        N, M, P = J.shape

    if P == 3:
        J = rgb2gray(J.astype(float))

    se = disk(3)
    Re = dilation(R, se)
    E = R - binary_erosion(R)

    #In the original implementation of Balu Matlab they used LoG edge detection.
    #Here we used edge detection via Canny algorithm. Then we used 0.5*sig because
    #that way we get similar results to Matlab version using edge command
    #L = canny(J, sigma=sig)
    L = edge_LoG(J, sigma=sig)
    L = np.logical_or(np.logical_and(L, Re), E)
    W = remove_small_objects(L, min_size=Amin, connectivity=2)
    F = label(np.logical_not(W), 4)
    m = int(F.max())
    for i in range(1, m + 1):
        ii, jj = np.where(F == i)
        if (ii.size < Amin) or (ii.size > N*M / 6.0):
            W[ii, jj] = 1

    F = label(np.logical_not(W), 4)
    m = int(F.max())
    print('{0} segmented regions.\n'.format(m))
    return F, m

def edge_LoG(I, sigma):
    LoG = laplace(gaussian(I, sigma=sigma), ksize=3)
    thres = np.absolute(LoG).mean() * 1.0
    output = sp.zeros(LoG.shape)
    w = output.shape[1]
    h = output.shape[0]

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            patch = LoG[y - 1:y + 2, x - 1:x + 2]
            p = LoG[y, x]
            maxP = patch.max()
            minP = patch.min()
            if p > 0:
                zeroCross = True if minP < 0 else False
            else:
                zeroCross = True if maxP > 0 else False
            if ((maxP - minP) > thres) and zeroCross:
                output[y, x] = 1

    #FIXME: It is necesary to define if return the closing of the output or just the output
    #return binary_closing(output)
    return output