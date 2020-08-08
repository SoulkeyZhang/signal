# -*- coding: utf-8 -*-
"""
Created on Mon May 25 14:50:05 2020

@author: localuser
"""

import torch
import torch.nn as nn
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 定义常量
INPUT_SIZE = 1  # 定义输入的特征数
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
        
        self.lstm2 = nn.LSTM(
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
        
    def forward(self, x2):    # x1是输入的时域数据，x2是输入的fft数据

        r_out, (h_s, h_c) = self.lstm2(x2)   # 如果不导入h_s和h_c，默认每次都进行0初始化
                                          #  h_s和h_c表示每一个隐层的上一时间点输出值和输入细胞状态
                                          # h_s和h_c的格式均是(num_layers * num_directions, batch, HIDDEN_SIZE)
        output = self.hidden_out(r_out)
        return output[:,-1,:]

if __name__ == '__main__':
    import torchvision
    import torchvision.datasets as dsets

    import torchvision.transforms as transforms

    import matplotlib.pyplot as plt
    
    BATCH_SIZE = 64
    print('start')
    #get the mnist dataset
    train_data = dsets.MNIST(root='./', train=True, transform=transforms.ToTensor(), download=False)
    test_data = dsets.MNIST(root='./', train=False, transform=transforms.ToTensor())
    test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255
    test_y = test_data.test_labels.numpy()[:2000]
    #use dataloader to batch input dateset
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    model = Lstm_Signal(28,64,num_layers=2,dropout=0.2,class_num=10).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    #define cross entropy loss function

    loss_func = nn.CrossEntropyLoss().to(device)
    
    for epoch in range(1):    
        for step, (b_x, b_y) in enumerate(train_loader):        #recover x as (batch, time_step, input_size)
            b_x = b_x.view(-1, 28, 28)

            output = model(b_x.cuda())
    
            loss = loss_func(output, b_y.cuda())
    
            optimizer.zero_grad()
    
            loss.backward()
    
            optimizer.step()
            
            if step % 50 == 0:            #train with rnn            
                test_output = model(test_x.cuda())            #loss function           
                pred_y = torch.max(test_output.cpu(), 1)[1].data.numpy()            #accuracy calculate      
                acc = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)           
                print('Epoch: ', (epoch), 'train loss: %.3f'%loss.cpu().data.numpy(), 'test acc: %.3f'%(acc))

        
        