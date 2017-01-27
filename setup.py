# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
setup(
  name = 'balu',
  packages=find_packages(),
  version = '0.1.15',
  include_package_data=True,
  description = 'Python implementation of Balu Toolbox by Domingo Mery (http://dmery.ing.puc.cl/index.php/balu/)',
  author = 'Diego Patiño',
  author_email = 'dapatinoco@unal.edu.com',
  url = 'https://github.com/dipaco/balu-python',
  download_url = 'https://github.com/dipaco/balu-python/releases/tag/v0.1.0',
  keywords = ['Computer Vision', 'Machine Learning', 'Image Processing'],
  classifiers = [],
  install_requires = [
      'numpy>=1.11.2,<2',
      'scipy>=0.18.1,<1',
      'matplotlib>=1.5.2,<2',
      'scikit-image>=0.12.3,<1',
      'scikit-learn>=0.18,<1',
  ],
)
