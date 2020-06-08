# -*- coding: utf-8 -*-
"""
Created on Sat May 23 10:35:10 2020

@author: localuser
"""

import numpy as np
from signalutility import Envelop
import torch


class SignalOfData:
    def __init__(self,data,sts,rpm,sr,piece_cnt=1,section_len = 0):
        self.rpm = rpm
        self.sample_rate = sr
        self.time_length = len(data)
        self.status = sts
        self.piece_cnt = piece_cnt
        self.time_domain = data  ##将时域信号进行normalization
        if self.piece_cnt == 1:
            
            self.section_len = self.time_length
        else:
            self.section_len = section_len
    def section(self):
        sec = self.time_domain[:self.piece_cnt*self.section_len].reshape((self.piece_cnt,self.section_len))
        return sec
    def compress(self,div = 2,need_envelope =True):
        ##用于将数据进行压缩，以满足后续操作需求
        if need_envelope:
            sec = self.envelope()
        else:
            sec = self.section()
        comp = np.zeros((sec.shape[0],sec.shape[1]//div))
        for i in range(comp.shape[1]):
            comp[:,i] = sec[:,i*div]
        return comp
    def envelope(self):
        secs = self.section()
        enve = np.zeros_like(secs)
        for i,sec in enumerate(secs):
            enve[i] = Envelop(sec)
        return enve
    
    def fft(self,need_envelope = True):
        if need_envelope:
            signal = self.envelope()
        else:
            signal = self.section()
        tempf = abs(np.fft.fft(signal))
        tempf[:,0] = 0  ##时域信号经过normalization后再fft会有一个很大的直流信号，不清零会对后续的lstm训练影响很大
        return tempf[:,:self.section_len//2]
    def copydata2ts(self):
        return torch.FloatTensor(self.fft())

