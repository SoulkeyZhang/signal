# -*- coding: utf-8 -*-
"""
Created on Sat May 23 10:35:10 2020

@author: localuser
"""

import numpy as np
from signalutility import Envelop,emd,split
import torch

SECOND_OF_MINUTE = 60



class SignalOfData:
    def __init__(self,data,sts,rpm,sr,need_envelope = True):
        self.rpm = rpm
        self.sample_rate = sr
        self.time_length = len(data)
        self.status = sts
        self.time_domain = np.squeeze(data) 
        self.time_domain_ts = torch.FloatTensor(data).squeeze()
        self.need_envelope = need_envelope
        self.section_len = 800  #计算每圈的采样个数，将每圈的采样数据作为一笔data
#        self.section_len = int(np.ceil(sr * SECOND_OF_MINUTE / rpm))*10  #计算每圈的采样个数，将每圈的采样数据作为一笔data

    def section(self):
        data_set = split(self.time_domain,self.section_len)                                                                           
        return data_set
    
    def envelope(self):
        secs = self.section()
        enve = [Envelop(sec) for sec in secs]
        return enve
    
    def emd(self):
        signals = [emd(sec) for sec in self.section()]
        return signals
    
    def fft(self,need_emd = False):
        if need_emd:
            signals = self.emd() #返回的是一个list
        else:
            signals = self.section()  #返回的是一个tuple
        temp = np.array([abs(np.fft.fft(signal-signal.mean())) for signal in signals])
        ret = temp[:,:self.section_len//2] ##/(self.section_len//2)

        return ret

    def copydata2ts(self,dsource):
        if dsource == 'fft':
            sigs = self.fft()
        if dsource == 'section':
            sigs = self.section()
        elif dsource == 'envelope':
            sigs = self.envelope()

        return torch.FloatTensor(sigs)
    
    def get_section_cnt(self):
        return len(self.section())
    
    
    
if __name__ == "__main__":
     signal1 = SignalOfData(inner_fd07_1730,1,1730,12000,False)
     fft = signal1.copydata2ts('fft')
     sec = signal1.copydata2ts('section')
     env = signal1.copydata2ts('envelope')
     
     npfft = np.array(fft)
     npsec = np.array(sec)
     npenv = np.array(env)
     
     import matplotlib.pyplot as plt
     plt.plot(npfft[0])

    