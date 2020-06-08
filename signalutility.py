# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:06:03 2020

@author: localuser
"""
from scipy import fftpack
import numpy as np

def Envelop(signal):
    hbSignal = fftpack.hilbert(signal)
    return np.sqrt(signal**2+hbSignal**2)

def normalArray(data):        
    mean_tr = np.mean(data)
    std_tr = np.std(data)
    normalizedata = (data -mean_tr)/std_tr
    if np.isnan(normalizedata).any():
        print("the data nan!")
        return data
    return normalizedata




