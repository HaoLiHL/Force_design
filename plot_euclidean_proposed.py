#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:56:25 2023

@author: lihao
"""
import numpy as np
import scipy as sp
import multiprocessing as mp
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.colors as plt_color
#from solvers.analytic_u_pp import Analytic
from utils import perm
from utils.desc import Desc

#task = np.load('/Users/HL/Desktop/Study/SFM/proposed_R_salic.npy',allow_pickle=True).item()
#task = np.load('/Users/HL/Desktop/Study/SFM/proposed_R_asp.npy',allow_pickle=True).item()
task=np.load('saved_model/task_h2_long.npy',allow_pickle=True).item()
#task=np.load('/Users/HL/Documents/GitHub/Force_design/saved_model/task_sali.npy',allow_pickle=True).item()

R = task['R_train'][0:10,:,:]


data_proposed =  np.array([[0.6493005159, 0.3335238106, -0.0001173225],
                [-0.6516737070, -0.0006357680, 0.0001145444],
                [1.2202008073, -0.9870460045, 0.0009573408],
                [0.0973857531, -1.0090107149, -0.0011697607]])

# data_proposed =  np.array([[-2.23705406, 2.687796, 0.59083897],
#                 [-2.32648441, 1.65978466, -0.27460047],
#                 [-1.75101906, 3.56717285, 0.02360984],
#                 [-1.9969224, 1.74574353, -1.23056591]])
# data_proposed = np.array([[-1.80679, 2.7824, -0.0121526],
#                  [-2.49502, 1.56503, -0.0111459],
#                  [-1.79752, 0.33387, -0.0111768],
#                  [-0.403997, 0.339724, -0.0123445],
#                  [0.317446, 1.56223, -0.0134477],
#                  [-0.394867, 2.79846, -0.0132553],
#                  [1.77367, 1.56683, -0.0148935],
#                  [0.244276, 4.01262, -0.0141552],
#                  [1.24339, 3.84217, -0.0148837],
#                  [2.33889, 0.303113, -0.0153897],
#                  [3.32679, 0.358603, -0.0164426],
#                  [2.49528, 2.61165, -0.0157335],
#                  [-2.33383, 3.73788, -0.0121251],
#                  [-3.58881, 1.56657, -0.0103269],
#                  [-2.3415, -0.613527, -0.0103224],
#                  [0.162941, -0.593149, -0.0124449]])


# R_test = np.array([-6.4300901476,	7.2331696223,	-0.5911664542,
# -5.9715435892,	8.555565608,	-0.5610225352,
# -4.6508442198,	8.8485158053,	-0.950795729,
# -3.7925281465,	7.8139228357,	-1.3511948229,
# -4.2511352389,	6.4893931244,	-1.3688682882,
# -5.5850764413,	6.1726550283,	-1.0009542423,
# -7.4542929189,	6.9931926667,	-0.3021570067,
# -6.6426765618,	9.3554913681,	-0.2387048525,
# -4.2850410444,	9.8785082166,	-0.9343369018,
# -2.7615561704,	8.009896485,	-1.6522815598,
# -3.3435317115,	5.5126558623,	-1.8315551262,
# -2.9153750375,	4.4799079277,	-0.9354378578,
# -3.1093061487,	4.5366674165,	0.2811105678,
# -2.2301457462,	3.4023997094,	-1.7159698343,
# -3.0114113718,	2.7910314003,	-2.199932976,
# -1.6417765253,	2.7704998751,	-1.0376995306,
# -1.5924774877,	3.825450286,	-2.5063696617,
# -6.109701237,	4.7939567118,	-1.0790541151,
# -7.4412451902,	4.7110920658,	-0.6676714226,
# -5.5049187571,	3.7717220042,	-1.4729343509,
# -7.7189163082,	3.7623559803,	-0.7493333001]).reshape(1,21,3)

R_new = np.append(R,data_proposed).reshape(R.shape[0]+1,4,-1)

desc = Desc(
        R.shape[1],
        interact_cut_off=task['interact_cut_off'],
        max_processes=None,
    )

R_desc_atom, R_d_desc_atom = desc.from_R(R_new,lat_and_inv=None,
                callback=None)

n_molecule = R_desc_atom.shape[0]


distance = np.zeros([n_molecule,n_molecule])
for i in range(n_molecule):
    for j in range(i):
        distance[i,j] = np.linalg.norm(R_desc_atom[i]-R_desc_atom[j])
        distance[j,i] =distance[i,j]
        


fig = plt.figure() #调用figure创建一个绘图对象
#(ax1, ax2) = plt.subplots(1, 2)
#(ax1, ax2) = fig.add_subplot(211)
ax = fig.add_subplot(111)
cax = ax.matshow(distance, cmap='bwr')  #绘制热力图，从-1到1
#ax.axis('off')
#cax = ax.matshow(correlations, cmap='Blues',vmin=np.min(k2), vmax=np.max(k2))  #绘制热力图，从-1到1
fig.colorbar(cax)  #将matshow生成热力图设置为颜色渐变条
#ticks = np.arange(0,9,1) #生成0-9，步长为1
#ax.set_xticks(ticks)  #生成刻度
#ax.set_yticks(ticks)
ax.set_title("Aspirin euclidean: distance between training and proposed")
#ax.set_xticklabels(names) #生成x轴标签
#ax.set_yticklabels(names)
#plt.imshow(I, cmap='gray');


fig.tight_layout()
#fig.savefig('/Users/HL/Desktop/Study/SFM/inverse_force/euclidean_distance_Salicylic.png', format='png',bbox_inches = 'tight',dpi=600,pad_inches = 0.2)
plt.show()