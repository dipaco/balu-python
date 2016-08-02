import numpy as np
from balu.FeatureExtraction import Bfx_fourier
from scipy.ndimage import imread
from matplotlib.pyplot import imshow, show

I = (imread('test_images/Lenna.png')[:, :, 1]).astype(float) / 256.0
R = np.ones(I.shape)
options = {'Mfourier': 64, 'Nfourier': 64, 'mfourier': 2, 'nfourier': 2}

print Bfx_fourier(I, options)