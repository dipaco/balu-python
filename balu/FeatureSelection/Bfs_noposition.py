# -*- coding: utf-8 -*-


def Bfs_noposition(f, fn):
    """ f_new, fn_new = Bfs_noposition(f, fn)

     Toolbox: Balu
         This procedure deletes the features related to the position.
         It deletes the following features:
            - center of grav i
            - center of grav j
            - Ellipse-centre i
            - Ellipse-centre j

     Example:
            import numpy as np
            from balu.ImagesAndData import balu_imageload
            from balu.ImageProcessing import Bim_segbalu
            from balu.FeatureExtraction import Bfx_basicgeo, Bfx_fitellipse
            from balu.FeatureSelection import Bfs_noposition
            from balu.InputOutput import Bio_printfeatures

            I = balu_imageload('testimg1.jpg')              # input image
            R, E, J = Bim_segbalu(I)                        # segmentation
            X1, Xn1 = Bfx_basicgeo(R)                       # basic geometric features
            X2, Xn2 = Bfx_fitellipse(R)                     # Ellipse features
            X3 = np.hstack((X1, X2))
            Xn3 = Xn1 + Xn2
            print '\nOriginal features\n'
            Bio_printfeatures(X3, Xn3)
            X4, Xn4 = Bfs_noposition(X3, Xn3)               # delete position features
            print '\nSelected features\n'
            Bio_printfeatures(X4, Xn4)

      See also Bfs_norotation, Bfs_nobackground.

     D.Mery, PUC-DCC, Nov. 2009
     http://dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    ii = list(range(len(fn)))
    s = ['center of grav i [px]   ',
         'center of grav j [px]   ',
         'Ellipse-centre i [px]   ',
         'Ellipse-centre j [px]   ']
    fn_new = list(fn)

    for name in s:
        if name in fn:
            ii.remove(fn.index(name))
            fn_new.remove(name)

    if len(ii) > 0:
        f_new = f[:, ii]

    return f_new, fn_new
