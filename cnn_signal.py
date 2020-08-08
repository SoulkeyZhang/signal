# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:43:41 2020

@author: localuser
"""

import torch.nn as nn
import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        


IN_CHANNEL = 1
OUT_CHANNEL = 16
KERNEL_SIZE = 64
STRIDE = 16
OUT_CHANNEL2 = 32
KERNEL_SIZE2 = 3
CLASS_NUM = 4
class Flatten(nn.Module):
    def forward(self,x):
        N,*other = x.size()
        return x.view(N,-1)


class CnnInSignal(nn.Module):
    def __init__(self,class_num = CLASS_NUM):
        super().__init__()
        self.__class_num = class_num
        self.cnn = nn.Sequential(
                    nn.Conv1d(in_channels=IN_CHANNEL,out_channels=OUT_CHANNEL,kernel_size=KERNEL_SIZE,stride=STRIDE),
#                    nn.BatchNorm1d(OUT_CHANNEL),
                    nn.ReLU(), 
                    nn.MaxPool1d(2,stride=1),
                    
                    nn.Conv1d(in_channels=OUT_CHANNEL,out_channels=OUT_CHANNEL2,kernel_size=KERNEL_SIZE2,stride=1),
#                    nn.BatchNorm1d(OUT_CHANNEL2),
                    nn.ReLU(),
                    nn.MaxPool1d(2,stride=2),
                    Flatten()
#                    nn.Dropout(0.2)
                )
        
    def forward(self,x):
        cnn_output = self.cnn(x)
        
        self.affine = nn.Sequential(
                                    nn.Linear(cnn_output.shape[1],32),
                                    nn.ReLU(),
                                    nn.Linear(32,self.__class_num),
                                    ).to(device)
        
        return self.affine(cnn_output)
              

