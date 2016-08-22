from scipy.io import loadmat
from os.path import dirname, realpath, join
from scipy.ndimage import imread

__all__ = ['balu_load', 'balu_imageload']


def balu_load(name):
    dir_path = dirname(realpath(__file__))
    return loadmat(join(dir_path, name + '.mat'))


def balu_imageload(name):
    dir_path = dirname(realpath(__file__))
    return imread(join(dir_path, name))
