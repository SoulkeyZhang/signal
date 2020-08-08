# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 22:40:24 2020

@author: localuser
"""
import numpy as np
from torch.utils.data import Dataset,random_split,TensorDataset
from signal_class import SignalOfData as Signal
import torch


TRAIN_RATE = 0.6
VALID_RATE = 0.2
TEST_RATE = 0.2 ##train_rate为训练数据占比

gpu_dtype = torch.cuda.FloatTensor
gpu_Itype = torch.cuda.IntTensor

def signal_data_set(datalist , rpm,sr):

    y = np.zeros(0,dtype = 'int')
    data_fft = np.zeros((0,400))
    
    for sts,data in enumerate(datalist):    
        
       data_cls = Signal(data,sts,rpm,sr,False)
       data_fft =np.vstack((data_cls.fft(),data_fft))
       y = np.append(np.array([sts] *data_cls.get_section_cnt()),y)

    data_set = TensorDataset(torch.from_numpy(data_fft), torch.from_numpy(y))   
    
    return data_set

def split_data_set(data_set):
    
    data_size = len(data_set)
    train_size = int(data_size * TRAIN_RATE)
    valid_size = int(data_size * VALID_RATE)
    test_size = data_size - train_size - valid_size
    
    data_set_train, data_set_valid,data_set_test = \
        random_split(data_set,(train_size,valid_size,test_size))
    
    return  data_set_train, data_set_valid,data_set_test 
    
class Data_set(Dataset):
    def __init__(self,X1,y):
        self.X1 = X1
        self.y = y
    
    def __getitem__(self,index):
        return self.X1[index], self.y[index]

    def __len__(self):
        return len(self.y)
    
    
if __name__ == '__main__':
    import torch
    
    X1 = torch.zeros((200,50))
    y = torch.zeros((200,))
    X1[0:100,15] = 1
    X1[100:,35] = 1
    X1 -= 0.5

    y[0:100] = 1
    data = Data_set(X1,y)
    print(len(data))
    
    datalist = [ball_fb07_1730,ball_fb07_1750,ball_fd07_1730]
    data_set = signal_data_set(datalist,1730,12000)
    print(len(data_set)) 
    
    data_train,data_valid,data_test = split_data_set(data_set)
    print(len(data_test))
    print(len(data_train))
    print(len(data_valid)) 
        
    