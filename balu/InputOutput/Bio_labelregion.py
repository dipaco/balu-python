# -*- coding: utf-8 -*-
import numpy as np
from skimage.color import rgb2gray
from warnings import filterwarnings
from balu.InputOutput import Bio_edgeview
from matplotlib.widgets import RadioButtons
from matplotlib.pyplot import close, imshow, axis, draw, colorbar, gca, plot, show, title, subplot, figure


def Bio_labelregion(I, L, c):
    """ d, D = Bio_labelregion(I, L, c)

     Toolbox: Balu
        User interface to label regions of an image.

        I is the original image (color or grayvalue).
        L is a labeled image that indicates the segmented regions of I.
        c is the maximal number of classes.
        d(i) will be the class number of region i.
        D is a binary image with the corresponding labels.

     Example:
        from balu.ImagesAndData import balu_imageload
        from balu.ImageProcessing import Bim_segmowgli
        from balu.InputOutput import Bio_labelregion

        I = balu_imageload('rice.png')
        I = I[0:70, 0:70]                                   # input image
        L, m = Bim_segmowgli(I, np.ones(I.shape), 40, 1.5)  # segmented image
        d, D = Bio_labelregion(I, L, 3)
        print d

     (c) Diego Patiño dapatinoco@unal.edu.co (2016)

     Original code:
     (c) GRIMA-DCCUC, 2011
     http://grima.ing.puc.cl
    """

    close('all')
    filterwarnings('ignore')

    if len(I.shape) == 3 and I.shape[2] == 3:
        J = rgb2gray(I.astype(float))
    else:
        J = I

    figure(1)

    n = L.max()
    d = np.zeros((n, 1))
    i = 1
    D = np.zeros(J.shape)
    labels = tuple(['Correct label'] + ['class {0}'.format(a + 1) for a in range(c)])

    data = {'n': n, 'd': d, 'i': i, 'c': c, 'D': D, 'J': J, 'L': L, 'labels': labels}

    # Create the radio buttons subplot
    colorstr = '0bgrcmykwbgrcmykwbgrcmykw'
    subplot(1, 3, 1)
    title('Select the class')
    radio = RadioButtons(gca(), labels=labels)
    [cir.set_radius(cir.get_radius() * 0.5) for cir in radio.circles]
    [t.set_color(colorstr[i]) for i, t in enumerate(radio.labels)]

    subplot(1, 3, 3)
    if len(I.shape) == 3:
        imshow(I)
    else:
        imshow(I, cmap='gray')
    title('Labeled regions')

    def next_s(label):
        return mark_object(label, data)

    radio.on_clicked(next_s)
    next_s(labels[0])

    show()
    figure(1)
    subplot(1, 2, 2)
    title('Class labels')
    imshow(D)
    colorbar().set_ticks(list(range(c + 1)))
    subplot(1, 2, 1)
    Bio_edgeview(J, np.zeros(J.shape), np.array([1, 1, 0]), show_now=False)
    title('Original image')
    show()
    return d, D


def mark_object(label, data):
    colorstr = 'bgrcmykwbgrcmykwbgrcmykw'

    n = data['n']
    d = data['d']
    i = data['i']
    c = data['c']
    D = data['D']
    J = data['J']
    L = data['L']
    labels = data['labels']

    if i <= n:
        ii, jj = np.where(L == i)

        if label == labels[0]:
            i = max(i - 2, 0)
        else:
            r = labels.index(label)
            d[i - 1] = r
            D[ii, jj] = r
            x1 = jj.max() + 1
            x2 = jj.min() - 1
            y1 = ii.max() + 1
            y2 = ii.min() - 1
            subplot(1, 3, 3)
            plot(np.array([x1, x1, x2, x2, x1]), np.array([y1, y2, y2, y1, y1]), color=colorstr[r - 1])
            axis([0, J.shape[1], 0, J.shape[0]])
            gca().invert_yaxis()
            draw()

        if i == n:
            close()

        i += 1
        data['i'] = i

    subplot(1, 3, 1)
    title('Region {0}/{1}: Class label? (1... {2}):'.format(i, n, c))
    draw()

    subplot(1, 3, 2)
    R = np.zeros(J.shape)
    kk, pp = np.where(L == i)
    R[kk, pp] = 1
    title('Class label of\nyellow region?')
    Bio_edgeview(J, R, np.array([1, 1, 0]), show_now=False)
    draw()


