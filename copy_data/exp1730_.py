# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:18:09 2020

@author: localuser
"""

import signal_class as sc
import lstm_class4signal as lc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from cnn_signal import CnnInSignal
from torch.utils.data import random_split,RandomSampler,DataLoader,WeightedRandomSampler
from data_set import Data_set


RPM_DEFAULT = 1000
SAMPLE_RATE_DEFAULT = 100000
TRAIN_RATE = 0.6
VALID_RATE = 0.2
TEST_RATE = 0.2 ##train_rate为训练数据占比
LEARNING_RATE = 1e-4
EPOCH_CYCLE= 20
BATCH_NUM = 20
EPOCH_BATCH_NUM = 800
LSTM_DATA_LEN = 200 #只取fft的前面部分，输入到lstm中。

gpu_dtype = torch.cuda.FloatTensor
gpu_Itype = torch.cuda.IntTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        



class Expriment:
    def __init__(self,data_list = [],
                 rpm = RPM_DEFAULT,
                 sample_rate = SAMPLE_RATE_DEFAULT,
                 need_envelope = True):
        self.__rpm = rpm
        self.__sample_rate = sample_rate
        self.__type_cnt = len(data_list) ##表示实验数据的工况种数,工况类型为输入的数据列表的index
        self.__is_first_train = {"lstm":True,"cnn":True}
        self.__signal_class_list = []
        self.__mode = {}

        for type_status, data_of_type in enumerate(data_list):
            sig_cls = sc.SignalOfData(data_of_type,type_status,rpm,sample_rate,need_envelope)   ##一次采集信号的实例
            self.__signal_class_list.append(sig_cls)

    @classmethod
    def mdata_set(cls,data_list,
                 rpm = RPM_DEFAULT,
                 sample_rate = SAMPLE_RATE_DEFAULT,
                 need_envelope = True):
        
        y = torch.IntTensor([])
        data_fft = torch.FloatTensor([])
        data_tdomain = torch.FloatTensor([])        

        for sts,data in enumerate(data_list):    
            
           data_cls = sc.SignalOfData(data,sts,rpm,sample_rate,need_envelope)
           data_fft = torch.cat((data_fft ,data_cls.copydata2ts(dsource='fft')))
           data_tdomain = torch.cat((data_tdomain,data_cls.copydata2ts(dsource='section')))
           y = torch.cat((y,torch.IntTensor(np.ones(data_cls.get_section_cnt())*sts)))
        data_set = Data_set(data_tdomain.type(gpu_dtype),data_fft.type(gpu_dtype),torch.zeros(len(y)).type(gpu_dtype),y.type(gpu_Itype))   
        return data_set
        
        
        
    def copy_data2ts(self):
        
        y = torch.IntTensor([])
        data_fft = torch.FloatTensor([])
        data_tdomain = torch.FloatTensor([])
        data_status_cnt = np.zeros(self.__type_cnt)
        
        for type_status, signal_of_type in enumerate(self.__signal_class_list):
            data_fft = torch.cat((data_fft ,signal_of_type.copydata2ts(dsource='fft')[:,:LSTM_DATA_LEN]))
            data_tdomain = torch.cat((data_tdomain,signal_of_type.copydata2ts(dsource='section')))
            y = torch.cat((y,torch.IntTensor(np.ones(signal_of_type.get_section_cnt())*type_status)))
            data_status_cnt[type_status] = signal_of_type.get_section_cnt()
            
        weight = torch.zeros(len(y))
        cursor = 0
        for cnt in data_status_cnt:
            weight[int(cursor):int(cursor+cnt)] = 100/cnt
            cursor += cnt       
            
        self.data_set = Data_set(data_tdomain.type(gpu_dtype),data_fft.type(gpu_dtype),weight.type(gpu_dtype),y.type(gpu_Itype))    
        data_set_size = len(self.data_set)
        train_size = int(data_set_size *TRAIN_RATE)
        valid_size = int(data_set_size * VALID_RATE)
        test_size = data_set_size - train_size - valid_size
        self.data_set_train, self.data_set_valid,self.data_set_test = random_split(self.data_set,(train_size,valid_size,test_size))
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        (x1,x2,self.weight),y = self.data_set_train[:]

            
    def __mode_init(self,nn_mode):
        self.copy_data2ts()
        self.__loss_fn = nn.CrossEntropyLoss().type(gpu_dtype)
        self.__is_first_train[nn_mode] = False
        if nn_mode == 'lstm':
           self.__mode[nn_mode] = lc.Lstm_Signal(class_num=self.__type_cnt).to(device)
        elif nn_mode == 'cnn':
           self.__mode[nn_mode] = CnnInSignal(self.__type_cnt).to(device)
     

    def __lstm_check_acc(self,mode_name,data_set):
        mode = self.__mode[mode_name]
        mode.eval()        
        num_correct=0           
        (x1_var,x2_var,_),y = data_set[:]
        scorn = mode(x2_var)
        if mode_name == 'lstm':
            pre = scorn[:,-1,:].argmax(1)            
        elif mode_name  == 'cnn':
            pre = scorn.argmax(1)
            
        pre = pre.type(gpu_Itype)
        num_correct += (y==pre).sum()

        return float(num_correct)/len(pre)
    
    def __model_train(self,mode_name,floss,foptim,epoch = 20):
        prtloss = []
        prtacc = []
        datalist = [i for i in range(self.train_size)]
        np.random.shuffle(datalist)
        mode = self.__mode[mode_name]
        for i in range(epoch):
            mode.train()
       #     sample = WeightedRandomSampler(weights=self.weight,num_samples=len(self.data_set_train))
            sample = WeightedRandomSampler(weights=self.weight,num_samples=EPOCH_BATCH_NUM)
            data_loader = DataLoader(self.data_set_train,BATCH_NUM,sampler=sample,drop_last=True)
            for train_batch in data_loader:
                (X_t, X_f,_), y_train = train_batch
                y_train = y_train.long()
                scorn = mode(X_f)
                if mode_name == 'lstm':
                    loss = floss(scorn[:,-1,:],y_train)
                elif mode_name == 'cnn':
                    loss = floss(scorn,y_train)
                loss.backward()
                foptim.step()
 
            if i %10 ==0:                
                print('j =%d , loss =%.4f' %(i+1, loss.item())) 
                prtloss.append(loss.item())       
    
            prtacc.append(self.__lstm_check_acc(mode_name,self.data_set_valid))
        print('acc rate = %.4f' %(prtacc[-1]))
        plt.plot(prtacc)  
        pass
       


        
    def train(self,learning_rate = LEARNING_RATE,epoch = EPOCH_CYCLE,nn_mode = 'lstm'):
        if self.__is_first_train[nn_mode]:
            self.__mode_init(nn_mode)
        optimizer =optim.Adam(self.__mode[nn_mode].parameters(),lr=learning_rate)    
        self.__model_train(nn_mode,self.__loss_fn,optimizer,epoch)  
        return 


    def lstm_mode_check(self,set_check = None):
        if self.__is_first_train['lstm']:
            print("mode not ready!")
            return 
        
        if set_check == None:
            set_check = self.data_set_valid

 
        self.__mode['lstm'].eval()

        (x1_var,x2_var,_),y = set_check[:]
        correct_cnt = np.zeros(self.__type_cnt)   
        type_amount = np.zeros(self.__type_cnt)   
        scorn = self.__mode['lstm'](x2_var)
        pre = scorn[:,-1,:].argmax(1)
        pre = pre.type(gpu_Itype) 

        for i in range(len(pre)):
            correct_cnt[y[i]] += (pre[i]==y[i])
            type_amount[y[i]] +=1

        for i in range(self.__type_cnt):
            print(f'{i} acc = {correct_cnt[i]/type_amount[i]}')
                      
        
data_exp = [inner_fd07_1730,outer_fd07_1730]          
      
exp = Expriment(data_exp,rpm = 1730,sample_rate=12000,need_envelope= False) 
exp.train(learning_rate=3e-4,epoch=50,nn_mode='cnn')