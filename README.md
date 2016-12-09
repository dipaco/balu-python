# balu-python
Python implementation of Balu Toolbox by Domingo Mery (http://dmery.ing.puc.cl/index.php/balu/)

## Dependencies:
- numpy >= 1.11.2
- scipy >= 0.18.1
- matplotlib >= 1.5.2
- scikit-image >= 0.12.3
- scikit-learn >= 0.18

## Installation:
### Mac OS:
1. Download and install `pip` if you haven't:

 - curl https://bootstrap.pypa.io/ez_setup.py -o - | sudo python
    
 - sudo easy_install pip
    
   Usually, if you use virtual environment step (1) is not required.
    
2. Install `balu-python` using pip:
    
 - pip install balu
 
### Linux:
    
## NOTES:
- We recommend to use a virtual environment to isolate the library from the system installed packages. 
- If you use a virtual environment it is recommended to run in python the command `matplotlib.use('TkAgg')` before importing any function from matplotlib.pyplot. (See here)[http://matplotlib.org/faq/virtualenv_faq.html]
