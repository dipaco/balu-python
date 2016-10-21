# -*- coding: utf-8 -*-
from skimage.filters import threshold_otsu
from .Bim_rgb2hcm import Bim_rgb2hcm
from .Bim_morphoreg import Bim_morphoreg


def Bim_segbalu(I, p=-0.05):
    """ R, E, J = Bim_segbalu(I, p)

     Toolbox: Balu
      Segmentation of an object with homogeneous background.

      I: input image
      p: threshold (default: p=-0.05) with p between -1 and 1.
          A positive value is used to dilate the segmentation,
          the negative to erode.
      R: binary image of the object
      E: binary image of the edge of the object
      J: high contrast image of I.

      See details in:
      Mery, D.; Pedreschi, F. (2005): Segmentation of Colour Food Images using
      a Robust Algorithm. Journal of Food Engineering 66(3): 353-360.

      Example:
            from balu.ImagesAndData import balu_imageload
            from balu.ImageProcessing import Bim_segbalu
            from matplotlib.pyplot import figure, imshow, title, show

            I = balu_imageload('testimg1.jpg')
            R, E, J = Bim_segbalu(I)
            figure(1)
            imshow(I), title('test image')
            figure(2)
            imshow(R, cmap='gray'), title('segmented image')
            show()

         Repeat this examples for images testimg2, testimg3 and testimg4. Last
         test image requires R = Bim_segbalu(I,-0.1) for better results.

     See also Bim_segmowgli, Bim_segotsu, Bim_segkmeans, Bio_segshow.

     D.Mery, PUC-DCC, Apr. 2008-2010
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    J = Bim_rgb2hcm(I.astype(float) / 256.0)
    t = threshold_otsu(J)
    R, E = Bim_morphoreg(J, t+p)
    return R, E, J
