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


RPM_DEFAULT = 1000
SAMPLE_RATE_DEFAULT = 100000
PIECE_CNT = 10
SECTION_LEN = 12000
TRAIN_RATE = 0.6
VALID_RATE = 0.2
TEST_RATE = 0.2 ##train_rate为训练数据占比
LEARNING_RATE = 1e-4
EPOCH_CYCLE= 20
BATCH_NUM = 20

gpu_dtype = torch.cuda.FloatTensor
gpu_Itype = torch.cuda.IntTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        



class Expriment:
    def __init__(self,data_list = [],
                 rpm = RPM_DEFAULT,
                 sample_rate = SAMPLE_RATE_DEFAULT,
                 piece_cnt = PIECE_CNT,section_len = SECTION_LEN,need_envelope = True):
        self.__rpm = rpm
        self.__sample_rate = sample_rate
        self.__type_cnt = len(data_list) ##表示实验数据的工况种数,工况类型为输入的数据列表的index
        self.__piece_cnt = piece_cnt
        self.__section_len = section_len
        self.__need_envelope = need_envelope
        self.__is_first_train = {"lstm":True,"cnn":True}
        self.__signal_class_list = []
        self.__mode = {}

        for type_status, data_of_type in enumerate(data_list):
            sig_cls = sc.SignalOfData(data_of_type,type_status,rpm,sample_rate,piece_cnt,section_len)   ##一次采集信号的实例
            self.__signal_class_list.append(sig_cls)


    def __copy_data2ts(self):
        data_train = np.zeros((self.__piece_cnt*self.__type_cnt,self.__section_len//2,2))
        y_train = np.zeros(self.__piece_cnt*self.__type_cnt,dtype = 'int')
        
        for type_status, signal_of_type in enumerate(self.__signal_class_list):
            data_train[type_status*self.__piece_cnt:(type_status+1)*self.__piece_cnt,:,0] = signal_of_type.fft(self.__need_envelope)
            data_train[type_status*self.__piece_cnt:(type_status+1)*self.__piece_cnt,:,1] = signal_of_type.compress(2,self.__need_envelope)
            y_train[type_status*self.__piece_cnt:(type_status+1)*self.__piece_cnt] = signal_of_type.status
            
        lt = np.arange(self.__piece_cnt*self.__type_cnt)        
        np.random.shuffle(lt)  ##将数据洗牌打乱
        self.__train_num = int(self.__piece_cnt*self.__type_cnt*TRAIN_RATE)
        self.__valid_num = int(self.__piece_cnt*self.__type_cnt*VALID_RATE)
        self.__test_num = int(self.__piece_cnt*self.__type_cnt*TEST_RATE)
        self.data_train = data_train[lt]
        self.y_train = y_train[lt]        
        self.data_train_ts = torch.FloatTensor(data_train[lt])
        self.y_train_ts = torch.IntTensor(y_train[lt])


    def __mode_init(self,nn_mode):
        self.__copy_data2ts()
        self.__loss_fn = nn.CrossEntropyLoss().type(gpu_dtype)
        self.__lstm_data_train_ts = self.data_train_ts
        self.__is_first_train[nn_mode] = False
        if nn_mode == 'lstm':
           self.__mode[nn_mode] = lc.Lstm_Signal(class_num=self.__type_cnt).to(device)
        elif nn_mode == 'cnn':
           self.__mode[nn_mode] = CnnInSignal(self.__type_cnt).to(device)
     

    def __lstm_check_acc(self,mode_name,data_set,y_set):
        mode = self.__mode[mode_name]
        mode.eval()        
        num_correct=0           
        x_var = data_set.type(gpu_dtype)
        scorn = mode(x_var)
        if mode_name == 'lstm':
            pre = scorn[:,-1,:].argmax(1)            
        elif mode_name  == 'cnn':
            pre = scorn.argmax(1)
            
        pre = pre.type(gpu_Itype)
        y_gpu_ts = y_set.type(gpu_Itype)
        num_correct += (y_gpu_ts==pre).sum()

        return float(num_correct)/len(pre)
    
    
    def __mode_train(self,mode_name,floss,foptim,epoc =1):
        datalist = [i for i in range(self.__train_num)]
        data_valid_list = [i for i in range(self.__train_num,self.__train_num+self.__valid_num)]
        np.random.shuffle(datalist)
        prtloss = []
        prtacc = []
        mode = self.__mode[mode_name]
        for i in range(epoc):
 ##           print('the epoc ',i+1)
            mode.train()
            for j in range(self.__train_num // BATCH_NUM):
                x_var =self.__lstm_data_train_ts[datalist[BATCH_NUM*j:BATCH_NUM*(j+1)]].type(gpu_dtype)
                y_var = self.y_train_ts[datalist[BATCH_NUM*j:BATCH_NUM*(j+1)]].type(gpu_dtype).long()                             
                scorn = mode(x_var)
                foptim.zero_grad()
                if mode_name == 'lstm':
                    loss = floss(scorn[:,-1,:],y_var)
                elif mode_name == 'cnn':
                    loss = floss(scorn,y_var)
                loss.backward()
                foptim.step()

            if i %10 ==0:                
                print('j =%d , loss =%.4f' %(i+1, loss.item())) 
                prtloss.append(loss.item())
    
            prtacc.append(self.__lstm_check_acc(mode_name,self.__lstm_data_train_ts[data_valid_list], self.y_train_ts[data_valid_list]))
        print('acc rate = %.4f' %(prtacc[-1]))
        plt.plot(prtacc)    


        
    def train(self,learning_rate = LEARNING_RATE,epoch = EPOCH_CYCLE,nn_mode = 'lstm'):
        if self.__is_first_train[nn_mode]:
            self.__mode_init(nn_mode)
        optimizer =optim.Adam(self.__mode[nn_mode].parameters(),lr=learning_rate)    
        self.__mode_train(nn_mode,self.__loss_fn,optimizer,epoch)  
        return 


    def lstm_mode_check(self,x_set = None,y_set = None):
        if self.__is_first_train['lstm']:
            print("mode not ready!")
            return 

        if x_set == None:
            data_valid_list = [i for i in range(self.__train_num,self.__train_num+self.__valid_num)]
            x_set = self.__lstm_data_train_ts[data_valid_list]
            y_set = self.y_train_ts[data_valid_list]

        self.__mode['lstm'].eval()
        num_correct=0
        correct_cnt = np.zeros(self.__type_cnt)   
        type_amount = np.zeros(self.__type_cnt)   
        x_var = x_set.type(gpu_dtype)
        scorn = self.__mode['lstm'](x_var)
        pre = scorn[:,-1,:].argmax(1)
        pre = pre.type(gpu_Itype)
        y_gpu_ts = y_set.type(gpu_Itype)
        num_correct += (y_gpu_ts==pre).sum()

        for i in range(len(pre)):
            correct_cnt[y_set[i]] += (pre[i]==y_gpu_ts[i])
            type_amount[y_set[i]] +=1

        for i in range(self.__type_cnt):
            print(f'{i} acc = {correct_cnt[i]/type_amount[i]}')
                      
        
        
        