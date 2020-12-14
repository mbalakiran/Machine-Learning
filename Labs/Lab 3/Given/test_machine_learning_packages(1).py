# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:59:53 2020

@author: hfl
"""
#############################################################################
# This is to test the installation of the required packages for machine
# learning
#############################################################################
# scipy
import scipy

print("scipy: %s" % scipy.__version__)
# numpy
import numpy

print("numpy: %s" % numpy.__version__)
# matplotlib
import matplotlib

print("matplotlib: %s" % matplotlib.__version__)
# pandas
import pandas

print("pandas: %s" % pandas.__version__)

# scikit-learn
# if the system has no sklearn you can install it from the anaconda prompt via
# conda install scikit-learn
import sklearn

print("sklearn: %s" % sklearn.__version__)
