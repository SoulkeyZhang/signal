# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:43:41 2020

@author: localuser
"""

import torch.nn as nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        


IN_CHANNEL = 2
OUT_CHANNEL = 30
KERNEL_SIZE = 10
OUT_CHANNEL2 = 50
KERNEL_SIZE2 = 5
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
                    nn.Conv1d(in_channels=IN_CHANNEL,out_channels=OUT_CHANNEL,kernel_size=KERNEL_SIZE,stride=1),
                    nn.MaxPool1d(KERNEL_SIZE,stride=1),
                    nn.ReLU(),  ##这里需要输入的数据长度，暂时没想好怎么传递，这里直接给的立即数，需要修改
                    nn.BatchNorm1d(OUT_CHANNEL),
                    nn.Conv1d(in_channels=OUT_CHANNEL,out_channels=OUT_CHANNEL2,kernel_size=KERNEL_SIZE2,stride=1),
                    nn.MaxPool1d(KERNEL_SIZE2,stride=KERNEL_SIZE2),
                    nn.ReLU(),
                    nn.BatchNorm1d(OUT_CHANNEL2),
                    Flatten()
                )
        
    def forward(self,x):
        x_T = x.transpose(1,2)
        cnn_output = self.cnn(x_T)
        self.affine = nn.Linear(cnn_output.shape[1],self.__class_num).to(device) ##这里由于模型的尺寸需要输入的数据信息，所以不能在构造的时候初始化，所以需要手动和构造保持一致性
        return self.affine(cnn_output)
              

