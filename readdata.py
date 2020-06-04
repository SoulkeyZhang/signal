# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:27:14 2020

@author: localuser
"""

import scipy.io as sio
import torch
import numpy as  np


import signalutility as su

###############################Read files##########################################
file = '1730/Normal.mat'
datanor_1730 = sio.loadmat(file)

file = '1730/0.007-Ball.mat'
data07ball_1730 = sio.loadmat(file)
file = '1730/0.007-InnerRace.mat'
data07i_1730 = sio.loadmat(file)
file = '1730/0.007-OuterRace6.mat'
data07o_1730 = sio.loadmat(file)

file = '1730/0.014-Ball.mat'
data14ball_1730 = sio.loadmat(file)
file = '1730/0.014-InnerRace.mat'
data14i_1730 = sio.loadmat(file)
file = '1730/0.014-OuterRace6.mat'
data14o_1730 = sio.loadmat(file)

file = '1730/0.021-Ball.mat'
data21ball_1730 = sio.loadmat(file)
file = '1730/0.021-InnerRace.mat'
data21i_1730 = sio.loadmat(file)
file = '1730/0.021-OuterRace6.mat'
data21o_1730 = sio.loadmat(file)
###############################Read files##########################################
file = '1750/Normal.mat'
datanor_1750 = sio.loadmat(file)

file = '1750/0.007-Ball.mat'
data07ball_1750 = sio.loadmat(file)
file = '1750/0.007-InnerRace.mat'
data07i_1750 = sio.loadmat(file)
file = '1750/0.007-OuterRace6.mat'
data07o_1750 = sio.loadmat(file)

file = '1750/0.014-Ball.mat'
data14ball_1750 = sio.loadmat(file)
file = '1750/0.014-InnerRace.mat'
data14i_1750 = sio.loadmat(file)
file = '1750/0.014-OuterRace6.mat'
data14o_1750 = sio.loadmat(file)

file = '1750/0.021-Ball.mat'
data21ball_1750 = sio.loadmat(file)
file = '1750/0.021-InnerRace.mat'
data21i_1750 = sio.loadmat(file)
file = '1750/0.021-OuterRace6.mat'
data21o_1750 = sio.loadmat(file)

###############################Read files##########################################
file = '1772/Normal.mat'
datanor_1772 = sio.loadmat(file)

file = '1772/0.007-Ball.mat'
data07ball_1772 = sio.loadmat(file)
file = '1772/0.007-InnerRace.mat'
data07i_1772 = sio.loadmat(file)
file = '1772/0.007-OuterRace6.mat'
data07o_1772 = sio.loadmat(file)

file = '1772/0.014-Ball.mat'
data14ball_1772 = sio.loadmat(file)
file = '1772/0.014-InnerRace.mat'
data14i_1772 = sio.loadmat(file)
file = '1772/0.014-OuterRace6.mat'
data14o_1772 = sio.loadmat(file)

file = '1772/0.021-Ball.mat'
data21ball_1772 = sio.loadmat(file)
file = '1772/0.021-InnerRace.mat'
data21i_1772 = sio.loadmat(file)
file = '1772/0.021-OuterRace6.mat'
data21o_1772 = sio.loadmat(file)



################################Read Datas#########################################
normal_de_1730 = su.normalArray(datanor_1730['X100_DE_time'])
normal_fe_1730 = su.normalArray(datanor_1730['X100_FE_time'])
ball_ff07_1730 = su.normalArray(data07ball_1730['X121_FE_time'])
ball_fd07_1730 = su.normalArray(data07ball_1730['X121_DE_time'])
ball_fb07_1730 = su.normalArray(data07ball_1730['X121_BA_time'])
ball_ff14_1730 = su.normalArray(data14ball_1730['X188_FE_time'])
ball_fd14_1730 = su.normalArray(data14ball_1730['X188_DE_time'])
ball_fb14_1730 = su.normalArray(data14ball_1730['X188_BA_time'])
ball_ff21_1730 = su.normalArray(data21ball_1730['X225_FE_time'])
ball_fd21_1730 = su.normalArray(data21ball_1730['X225_DE_time'])
ball_fb21_1730 = su.normalArray(data21ball_1730['X225_BA_time'])
inner_ff07_1730 = su.normalArray(data07i_1730['X108_FE_time'])
inner_fd07_1730 = su.normalArray(data07i_1730['X108_DE_time'])
inner_fb07_1730 = su.normalArray(data07i_1730['X108_BA_time'])
inner_ff14_1730 = su.normalArray(data14i_1730['X172_FE_time'])
inner_fd14_1730 = su.normalArray(data14i_1730['X172_DE_time'])
inner_fb14_1730 = su.normalArray(data14i_1730['X172_BA_time'])
inner_ff21_1730 = su.normalArray(data21i_1730['X212_FE_time'])
inner_fd21_1730 = su.normalArray(data21i_1730['X212_DE_time'])
inner_fb21_1730 = su.normalArray(data21i_1730['X212_BA_time'])
outer_ff07_1730 = su.normalArray(data07o_1730['X133_FE_time'])
outer_fd07_1730 = su.normalArray(data07o_1730['X133_DE_time'])
outer_fb07_1730 = su.normalArray(data07o_1730['X133_BA_time'])
outer_ff14_1730 = su.normalArray( data14o_1730['X200_FE_time'])
outer_fd14_1730 = su.normalArray( data14o_1730['X200_DE_time'])
outer_fb14_1730 = su.normalArray( data14o_1730['X200_BA_time'])
outer_ff21_1730 = su.normalArray( data21o_1730['X237_FE_time'])
outer_fd21_1730 = su.normalArray( data21o_1730['X237_DE_time'])
outer_fb21_1730 = su.normalArray( data21o_1730['X237_BA_time'])
################################Read Datas#########################################
normal_de_1750 = su.normalArray(datanor_1750['X099_DE_time'])
normal_fe_1750 = su.normalArray(datanor_1750['X099_FE_time'])
ball_ff07_1750 = su.normalArray(data07ball_1750['X120_FE_time'])
ball_fd07_1750 = su.normalArray(data07ball_1750['X120_DE_time'])
ball_fb07_1750 = su.normalArray(data07ball_1750['X120_BA_time'])
ball_ff14_1750 = su.normalArray(data14ball_1750['X187_FE_time'])
ball_fd14_1750 = su.normalArray(data14ball_1750['X187_DE_time'])
ball_fb14_1750 = su.normalArray(data14ball_1750['X187_BA_time'])
ball_ff21_1750 = su.normalArray(data21ball_1750['X224_FE_time'])
ball_fd21_1750 = su.normalArray(data21ball_1750['X224_DE_time'])
ball_fb21_1750 = su.normalArray(data21ball_1750['X224_BA_time'])
inner_ff07_1750 = su.normalArray(data07i_1750['X107_FE_time'])
inner_fd07_1750 = su.normalArray(data07i_1750['X107_DE_time'])
inner_fb07_1750 = su.normalArray(data07i_1750['X107_BA_time'])
inner_ff14_1750 = su.normalArray(data14i_1750['X171_FE_time'])
inner_fd14_1750 = su.normalArray(data14i_1750['X171_DE_time'])
inner_fb14_1750 = su.normalArray(data14i_1750['X171_BA_time'])
inner_ff21_1750 = su.normalArray(data21i_1750['X211_FE_time'])
inner_fd21_1750 = su.normalArray(data21i_1750['X211_DE_time'])
inner_fb21_1750 = su.normalArray(data21i_1750['X211_BA_time'])
outer_ff07_1750 = su.normalArray(data07o_1750['X132_FE_time'])
outer_fd07_1750 = su.normalArray(data07o_1750['X132_DE_time'])
outer_fb07_1750 = su.normalArray(data07o_1750['X132_BA_time'])
outer_ff14_1750 = su.normalArray( data14o_1750['X199_FE_time'])
outer_fd14_1750 = su.normalArray( data14o_1750['X199_DE_time'])
outer_fb14_1750 = su.normalArray( data14o_1750['X199_BA_time'])
outer_ff21_1750 = su.normalArray( data21o_1750['X236_FE_time'])
outer_fd21_1750 = su.normalArray( data21o_1750['X236_DE_time'])
outer_fb21_1750 = su.normalArray( data21o_1750['X236_BA_time'])
################################Read Datas#########################################
normal_de_1772 = su.normalArray(datanor_1772['X098_DE_time'])
normal_fe_1772 = su.normalArray(datanor_1772['X098_FE_time'])
ball_ff07_1772 = su.normalArray(data07ball_1772['X119_FE_time'])
ball_fd07_1772 = su.normalArray(data07ball_1772['X119_DE_time'])
ball_fb07_1772 = su.normalArray(data07ball_1772['X119_BA_time'])
ball_ff14_1772 = su.normalArray(data14ball_1772['X186_FE_time'])
ball_fd14_1772 = su.normalArray(data14ball_1772['X186_DE_time'])
ball_fb14_1772 = su.normalArray(data14ball_1772['X186_BA_time'])
ball_ff21_1772 = su.normalArray(data21ball_1772['X223_FE_time'])
ball_fd21_1772 = su.normalArray(data21ball_1772['X223_DE_time'])
ball_fb21_1772 = su.normalArray(data21ball_1772['X223_BA_time'])
inner_ff07_1772 = su.normalArray(data07i_1772['X106_FE_time'])
inner_fd07_1772 = su.normalArray(data07i_1772['X106_DE_time'])
inner_fb07_1772 = su.normalArray(data07i_1772['X106_BA_time'])
inner_ff14_1772 = su.normalArray(data14i_1772['X170_FE_time'])
inner_fd14_1772 = su.normalArray(data14i_1772['X170_DE_time'])
inner_fb14_1772 = su.normalArray(data14i_1772['X170_BA_time'])
inner_ff21_1772 = su.normalArray(data21i_1772['X210_FE_time'])
inner_fd21_1772 = su.normalArray(data21i_1772['X210_DE_time'])
inner_fb21_1772 = su.normalArray(data21i_1772['X210_BA_time'])
outer_ff07_1772 = su.normalArray(data07o_1772['X131_FE_time'])
outer_fd07_1772 = su.normalArray(data07o_1772['X131_DE_time'])
outer_fb07_1772 = su.normalArray(data07o_1772['X131_BA_time'])
outer_ff14_1772 = su.normalArray( data14o_1772['X198_FE_time'])
outer_fd14_1772 = su.normalArray( data14o_1772['X198_DE_time'])
outer_fb14_1772 = su.normalArray( data14o_1772['X198_BA_time'])
outer_ff21_1772 = su.normalArray( data21o_1772['X235_FE_time'])
outer_fd21_1772 = su.normalArray( data21o_1772['X235_DE_time'])
outer_fb21_1772 = su.normalArray( data21o_1772['X235_BA_time'])