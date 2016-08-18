from scipy.io import loadmat
from os.path import dirname, realpath, join

__all__ = ['balu_load']


def balu_load(name):
    dir_path = dirname(realpath(__file__))
    return loadmat(join(dir_path, name + '.mat'))
