# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:06:03 2020

@author: localuser
"""
from scipy import fftpack
import numpy as np
import torch
from PyEMD import EMD,EEMD
from scipy import stats

def Envelop(signal):
    hbSignal = fftpack.hilbert(signal)
    if isinstance(signal,torch.Tensor):
        hbSignal = torch.Tensor(hbSignal)
    return np.sqrt(signal**2+hbSignal**2)

def normalArray(data):  
    return data     
    mean_tr = np.mean(data)
    return (data -mean_tr)
    std_tr = np.std(data)
    normalizedata = (data -mean_tr)/std_tr
    if np.isnan(normalizedata).any():
        print("the data nan!")
        return data
    return normalizedata


def emd(signal):    
    emd_cls = EEMD()
    imfs = emd_cls(signal)
    vkur = np.zeros(len(signal))
    temp = [imf for imf in imfs  if stats.kurtosis(imf) > 0]
    for imf in temp:
        vkur += imf
    hbSignal = abs(fftpack.hilbert(vkur))

    return hbSignal

def split(signal,section_len):
    section_cnt = len(signal)//section_len
    signal = signal[:section_cnt*section_len]
    return np.array(np.split(signal,section_cnt))
    