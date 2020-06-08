# -*- coding: utf-8 -*-
"""
Created on Mon May 25 14:50:05 2020

@author: localuser
"""

import torch
import torch.nn as nn
import numpy as np



# 定义常量
INPUT_SIZE = 2  # 定义输入的特征数
HIDDEN_SIZE = 15    # 定义一个LSTM单元有多少个神经元
DROP_RATE = 0.2    #  drop out概率
LAYERS = 2         # 有多少隐层，一个隐层一般放一个LSTM单元
MODEL = 'LSTM'     # 模型名字
CLASS_NUM = 4


class Lstm_Signal(nn.Module):
    def __init__(self,      
            input_size = INPUT_SIZE, 
            hidden_size = HIDDEN_SIZE, 
            num_layers = LAYERS,
            dropout = DROP_RATE,
            class_num = CLASS_NUM,
            batch_first = True    # 如果为True，输入输出数据格式是(batch, seq_len, feature)
                                  # 为False，输入输出数据格式是(seq_len, batch, feature)，)            
            ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size = input_size, 
            hidden_size = hidden_size, 
            num_layers = num_layers,
            dropout = dropout,
            batch_first = batch_first    # 如果为True，输入输出数据格式是(batch, seq_len, feature)
                                  # 为False，输入输出数据格式是(seq_len, batch, feature)，)
            )
        
        
        self.hidden_out = nn.Linear(hidden_size, class_num)  # 最后一个时序的输出接一个全连接层
        self.h_s = None
        self.h_c = None
        
    def forward(self, x):    # x是输入数据集
        r_out, (h_s, h_c) = self.lstm(x)   # 如果不导入h_s和h_c，默认每次都进行0初始化
                                          #  h_s和h_c表示每一个隐层的上一时间点输出值和输入细胞状态
                                          # h_s和h_c的格式均是(num_layers * num_directions, batch, HIDDEN_SIZE)
                                          # 如果是双向LSTM，num_directions是2，单向是1
        output = self.hidden_out(r_out)
        return output


