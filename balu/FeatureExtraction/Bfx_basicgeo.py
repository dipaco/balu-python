# -*- coding: utf-8 -*-
import numpy as np
from warnings import filterwarnings
from skimage.measure import regionprops, perimeter
from scipy.ndimage.morphology import distance_transform_cdt

def Bfx_basicgeo(R, *args):
    """X, Xn = Bfx_geobasic(R, options)

     Toolbox: Balu

     Standard geometric features of a binary image R.
     This function calls regionprops of Scikit image library.

     options['show'] = True display messages.

     X is the feature vector
     Xn is the list of feature names.

     See example:

     Example:
     from balu.ImagesAndData import balu_imageload
     I = balu_imageload('testimg1.jpg') #input image
     R = Bim_segbalu(I) # segmentation
     X, Xn = Bfx_basicgeo(R); # basic geometric features
     Bio_printfeatures(X, Xn)

     See also Bfx_fitellipse, Bfx_hugeo, Bfx_gupta, Bfx_flusser.

     (c) D.Mery, PUC - DCC, 2010
     http: // dmery.ing.puc.cl

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    if len(args) > 0:
        options = args[0]
    else:
        options = {'show': False}

    R = R.astype(bool)
    N = R.shape[0]
    filterwarnings('ignore')
    stats = regionprops(R.astype('uint8'))

    # Standard features
    # Perimeter
    if options['show']:
        print('--- extracting standard geometric features...')

    L8 = perimeter(R.astype('uint8'), neighbourhood=8)
    L4 = perimeter(R.astype('uint8'), neighbourhood=4)
    L = (3 * L4 + L8) / 4

    # Area
    A = stats[0].area

    # Roundness
    Roundness = 4 * A * np.pi / (L ** 2)

    # height & width
    [Ireg, Jreg] = np.where(R == 1)  # pixels in the region
    i_max = max(Ireg)
    i_min = min(Ireg)
    j_max = max(Jreg)
    j_min = min(Jreg)
    height = i_max - i_min + 1       # height
    width = j_max - j_min + 1        # width

    # Danielsson shape factor(see Danielsson, 1977)
    TD = distance_transform_cdt(R.astype('uint8'), metric='chessboard')
    dm = np.mean(TD[Ireg, Jreg])
    Gd = A / 9 / np.pi / dm ** 2

    X = np.array([[
        stats[0].centroid[0],
        stats[0].centroid[1],
        height,
        width,
        A,
        L,
        Roundness,
        Gd,
        stats[0].euler_number,
        stats[0].equivalent_diameter,
        stats[0].major_axis_length,
        stats[0].minor_axis_length,
        stats[0].orientation * 180.0 / np.pi,
        stats[0].solidity,
        stats[0].extent,
        stats[0].eccentricity,
        stats[0].convex_area,
        stats[0].filled_area]]
    );

    Xn = [
        'center of grav i [px]   ',
        'center of grav j [px]   ',
        'Height [px]             ',
        'Width [px]              ',
        'Area [px]               ',
        'Perimeter [px]          ',
        'Roundness               ',
        'Danielsson factor       ',
        'Euler Number            ',
        'Equivalent Diameter [px]',
        'MajorAxisLength [px]    ',
        'MinorAxisLength [px]    ',
        'Orientation [grad]      ',
        'Solidity                ',
        'Extent                  ',
        'Eccentricity            ',
        'Convex Area [px]        ',
        'Filled Area [px]        ']
    return X, Xn