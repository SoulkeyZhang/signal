# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:29:16 2020

@author: localuser
"""

import numpy as np
from torch.utils.data import WeightedRandomSampler,DataLoader,RandomSampler
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
import matplotlib.pyplot as plt


gpu_dtype = cuda.FloatTensor
gpu_Itype = cuda.IntTensor

LEARNING_RATE = 1e-4
EPOCH_CYCLE= 20
EPOCH_BATCH_NUM = 200
BATCH_NUM = 64

class Exprient:
    def __init__(self,data_set,mode):
        self.data_set = data_set
        self.train_set,self.valid_set,self.test_set = data_set
        self.train_size = len(self.train_set)
        self.__loss_func = nn.CrossEntropyLoss().type(gpu_dtype)
        x1,y = self.train_set[:]
        self.mode = mode
        
    def __train(self,optimizer,epoch = 20):
        prtloss = []
        prtacc = []

        for i in range(epoch):
            
            self.mode.train()
            sample = RandomSampler(self.train_set,replacement=True,num_samples=10000)
            data_loader = DataLoader(self.train_set,BATCH_NUM,sampler=sample,drop_last=True)
              
            for step, (X_f, y) in enumerate(data_loader):
                
                y = y.cuda().long()
                scorn = self.mode(X_f[:,:,np.newaxis].float().cuda())
                optimizer.zero_grad()
                loss = self.__loss_func(scorn,y)
                loss.backward()
                optimizer.step()
                if step % 50 == 0:
                    print('loss =%.4f' %( loss.item())) 
                prtloss.append(loss.item())                  
                
            print('j =%d , loss =%.4f' %(i+1, loss.item())) 
            prtacc.append(self.__lstm_check_acc(self.valid_set))
        print('acc rate = %.4f' %(prtacc[-1]))
        plt.plot(prtloss)  
        
    def train(self,learning_rate = LEARNING_RATE,epoch = EPOCH_CYCLE):
      
        optimizer =optim.Adam(self.mode.parameters(),lr=learning_rate)    
        self.__train(optimizer,epoch)         
        
    def __lstm_check_acc(self,data_set):

        self.mode.eval()        
        num_correct=0           
        x_var,y = data_set[:]
        scorn = self.mode(x_var[:,:,np.newaxis].float().cuda())

        pre = scorn.argmax(1)            
        pre = pre.type(gpu_Itype)
        num_correct += (y.cuda()==pre).sum()

        return float(num_correct)/len(pre)
    
    def check_set(self,data_set,class_num):

        self.mode.eval()  
        
        X,y = data_set.tensors
        correct_cnt = np.zeros(class_num)   
        type_amount = np.zeros(class_num)   

        scorn = self.mode(X[:,:,np.newaxis].float().cuda())

        pre = scorn.argmax(1)            
        pre = pre.type(gpu_Itype)

        for i in range(len(pre)):
            correct_cnt[y[i]] += (pre[i]==y.cuda()[i])
            type_amount[y[i]] +=1

        for i in range(class_num):
            print(f'sts{i} acc = {correct_cnt[i]/type_amount[i]}')        
    
    
if __name__ == '__main__':
    from cnn_signal import CnnInSignal
    from lstm_class4signal import Lstm_Signal as Lsmode
    import torch
    from data_set import split_data_set,signal_data_set
    import data_set
   
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
    datalist = [ball_fd07_1730,inner_fd07_1730,outer_fd07_1730,normal_de_1730]
    data_set = signal_data_set(datalist,1730,12000)
    data_train,data_valid,data_test = split_data_set(data_set)
    mode = Lsmode(1,128,num_layers=2,dropout=0.2,class_num=4).to(device)
    exp = Exprient((data_train,data_valid,data_test),mode)
    
    exp.train(learning_rate=1e-4,epoch=6) 
#    valid_set = signal_data_set([normal_de_1750,ball_fd07_1750,inner_fd07_1750,outer_fd07_1750],1750,12000)
#    exp.check_set(valid_set,4)