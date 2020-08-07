#!/usr/local/bin/python3.7
import matplotlib.pyplot as pylab

def set_rcparams():
    params = {'figure.figsize': (15, 5),
         'axes.labelsize': 14,
         'axes.titlesize': 14,
          'axes.titlesize': 18}
    pylab.rcParams.update(params)