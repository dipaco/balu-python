# -*- coding: utf-8 -*-
from scipy.io import loadmat
from os.path import dirname, realpath, join
from scipy.misc import imread

__all__ = ['balu_load', 'balu_imageload']


def balu_load(name):
    """ data = balu_load(data_filename)     #without extension

       Example:
          from balu.ImagesAndData import balu_load
          data = balu_load('datareal')         # data filename

    (c) Diego Patiño (dapatinoco@unal.edu.co) (2016)
    """
    dir_path = dirname(realpath(__file__))
    return loadmat(join(dir_path, name + '.mat'))


def balu_imageload(name):
    """ I = balu_imageload(image_filename)     #Image filename

       Example:
          from balu.ImagesAndData import balu_imageload
          I = balu_imageload(image_filename)         # Image filename

    (c) Diego Patiño (dapatinoco@unal.edu.co) (2016)
    """
    dir_path = dirname(realpath(__file__))
    return imread(join(dir_path, name))
