#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:07:41 2023

@author: lihao

"""

import numpy as np
import scipy as sp

task=np.load('saved_model/task_h2co.npy',allow_pickle=True).item()

data_initial = np.array([[-1.80673, 2.77777, 0.00194083],
                 [-2.54244, 1.57731, -0.0559721],
                 [-1.81219, 0.365089, 0.0192238],
                 [-0.400384, 0.333364, 0.0186005],
                 [0.336884, 1.56232, 0.0366634],
                 [-0.40434, 2.76748, 0.0558412],
                 [1.77871, 1.5174, -0.0272047],
                 [0.178311, 4.01824, 0.0033216],
                 [1.17744, 3.91418, -0.101221],
                 [2.354, 0.244263, -0.0119212],
                 [3.33003, 0.347375, -0.0660872],
                 [2.50323, 2.53921, -0.00890359],
                 [-2.27463, 3.78897, -0.123232],
                 [-3.66733, 1.5823, -0.00311385],
                 [-2.27342, -0.641083, 0.0338403],
                 [0.226967, -0.563954, -0.0469366]])

data_proposed = np.array([[-1.80679, 2.7824, -0.0121526],
                 [-2.49502, 1.56503, -0.0111459],
                 [-1.79752, 0.33387, -0.0111768],
                 [-0.403997, 0.339724, -0.0123445],
                 [0.317446, 1.56223, -0.0134477],
                 [-0.394867, 2.79846, -0.0132553],
                 [1.77367, 1.56683, -0.0148935],
                 [0.244276, 4.01262, -0.0141552],
                 [1.24339, 3.84217, -0.0148837],
                 [2.33889, 0.303113, -0.0153897],
                 [3.32679, 0.358603, -0.0164426],
                 [2.49528, 2.61165, -0.0157335],
                 [-2.33383, 3.73788, -0.0121251],
                 [-3.58881, 1.56657, -0.0103269],
                 [-2.3415, -0.613527, -0.0103224],
                 [0.162941, -0.593149, -0.0124449]])

data = np.array([[-1.8010790186, 2.7795208936, -0.0270775863],
        [-2.5027727123, 1.5717018612, -0.0137080685],
        [-1.8192913409, 0.3318819693, 0.0035645948],
        [-0.4243526388, 0.3147780814, 0.0030938459],
        [0.3115735621, 1.5275803120, -0.0138915639],
        [-0.3899435517, 2.7757036154, -0.0246676875],
        [1.7658860492, 1.5529422417, -0.0227590792],
        [0.2552640403, 3.9850459294, -0.0350842229],
        [1.2565357839, 3.8083710484, -0.0348558892],
        [2.3795606509, 0.3157243589, -0.0270001248],
        [3.3616747878, 0.4564837054, -0.0341226665],
        [2.4706920702, 2.6125442868, -0.0305999496],
        [-2.3116892595, 3.7441901479, -0.0379702830],
        [-3.5963370916, 1.5867447993, -0.0166279332],
        [-2.3798268320, -0.6055604036, 0.0170963188],
        [0.1282135011, -0.6274188471, 0.0194496850]])

data2 = np.array([[-1.8113508207, 2.7990201515, 0.0000000000],
                 [-2.5225398467, 1.5966769753, 0.0000000000],
                 [-1.8483305703, 0.3514846204, 0.0000000000],
                 [-0.4534209434, 0.3239157876, 0.0000000000],
                 [0.2921832532, 1.5305872847, 0.0000000000],
                 [-0.3999720073, 2.7841125457, 0.0000000000],
                 [1.7465465872, 1.5430917915, 0.0000000000],
                 [0.2556595069, 3.9877008557, 0.0000000000],
                 [1.2553777317, 3.8027780440, 0.0000000000],
                 [2.3503906176, 0.3009239570, 0.0000000000],
                 [3.3333474471, 0.4358023869, 0.0000000000],
                 [2.4596662317, 2.5971464446, 0.0000000000],
                 [-2.3147242268, 3.7675281604, 0.0000000000],
                 [-3.6161538299, 1.6224139673, 0.0000000000],
                 [-2.4181965847, -0.5805856386, 0.0000000000],
                 [0.0909374543,   -0.6229973342,    0.0000000000]])

import scipy.spatial
import sklearn as sk
from sklearn import metrics

task=np.load('saved_model/task_sali.npy',allow_pickle=True).item()

#a = scipy.spatial.distance.pdist(data, 'euclidean')
opt = metrics.pairwise_distances(data)
opt2 = metrics.pairwise_distances(data2)
initlal = metrics.pairwise_distances(data_initial)
initlal2 = metrics.pairwise_distances(task['R_train'][8,:,:])
proposed = metrics.pairwise_distances(data_proposed)



print(np.sum( (initlal - initlal2)**2))


R_difference =[]
F_loss = []
F_proposed = np.array([[[ 5.383550e-02,  3.760183e-01,  1.185800e-03],
                 [ 1.375530e-02,  6.494646e-01,  2.372000e-04],
                 [-1.726530e-01,  3.654647e-01, -4.743000e-04],
                 [-3.617887e-01,  7.363840e-02, -2.371600e-03],
                 [-3.865720e-02, -0.000000e+00,  3.913200e-03],
                 [ 2.952650e-02,  1.926931e-01,  1.186000e-04],
                 [ 5.383550e-02, -1.439566e-01,  9.486000e-04],
                 [ 2.658572e-01,  6.059460e-02, -1.541500e-03],
                 [ 3.403256e-01,  3.675990e-02, -2.845900e-03],
                 [-2.659758e-01, -3.448317e-01,  3.083100e-03],
                 [ 7.270162e-01, -0.000000e+00, -2.372000e-04],
                 [ 1.399250e-02, -2.709561e-01, -3.083100e-03],
                 [ 2.021795e-01,  4.706455e-01,  1.304400e-03],
                 [-4.909230e-02,  5.042037e-01,  1.304400e-03],
                 [-5.932575e-01,  4.823849e-01,  2.372000e-04],
                 [-2.190179e-01,  1.453795e-01, -1.778700e-03]]])

print( np.sum(F_proposed**2))

F_real = np.array([[[-0.0794488, 0.0757729, 0.210243],
                 [0.0535983, -0.1269996, -0.0570372],
                 [0.3092576, -0.2420225, -0.0337954],
                 [0.1490555, 0.1354188, 0.1597277],
                 [0.0004743, -0.0, 0.1515457],
                 [0.0257319, -0.0026088, -0.1638781],
                 [0.0601202, -0.0403173, -0.1248651],
                 [0.0624919, -0.1126513, -0.1431265],
                 [-0.1929302, -0.0596459, 0.0463649],
                 [0.0411474, -0.0175499, 0.2090572],
                 [0.1012676, -0.0, 0.0085378],
                 [-0.0235975, 0.0705553, 0.0021344],
                 [0.0009486, -0.0208701, -0.0342697],
                 [-0.1242722, 0.0983031, 0.0820576],
                 [-0.1711115, 0.0809904, -0.0026088],
                 [-0.2126146, 0.2735649, -0.3103248]]])

print( np.sum(F_real**2))
for i in range(task['R_train'].shape[0]):
    
    pair_dist_mat = metrics.pairwise_distances(task['R_train'][i,:,:])
    
    R_difference.append( np.sum((pair_dist_mat-opt)**2))
    F_loss.append(np.sum((task['F_train'][i,:,:])**2))


F_loss_1_iter =  61.732085749379
F_loss_2_iter =  24.765787579946302
F_loss_3_iter =  11.51920333718526

R_difference_our = np.sum((proposed-opt)**2)
F_difference_our = np.sum((proposed-opt)**2)
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

# Plot the first subplot (scatter plot)
ax1.scatter(np.arange(task['R_train'].shape[0]), np.log(R_difference))
ax1.axhline(y=np.log(R_difference_our), color='r', linestyle='--',label='the difference between simulator-opy and ML-opt')

# Set the plot title and axes labels
ax1.set_title("Difference of the pairwise Euclidean Distance Matrix")
ax1.set_xlabel("the index of R_train")
ax1.set_ylabel("Euclidean difference between R_train and R_opt (log)")
ax1.legend(loc='upper left')

# Plot the second subplot (line plot)
ax2.scatter(np.arange(task['F_train'].shape[0]), (F_loss),label='the loss of training sample')
ax2.axhline(y=(np.sum(F_proposed**2)), color='r', linestyle='--',label='the loss of simulator-opt')
ax2.axhline(y=(np.sum(F_real**2)), color='y', linestyle='-.',label='the loss of ML-opt')
ax2.axhline(y=F_loss_1_iter, color='b', linestyle='-.',label='the loss after 1st iter')
ax2.axhline(y=F_loss_2_iter, color='b', linestyle='-.',label='the loss after 2st iter')
ax2.axhline(y=F_loss_3_iter, color='b', linestyle='-.',label='the loss after 3st iter')

# Set the plot title and axes labels
ax2.set_title("Difference of the pairwise Euclidean Distance Matrix")
ax2.set_xlabel("the index of F_train")
ax2.set_ylabel("The L-2 Loss of the real F_train ")
ax2.legend(loc='upper left')


# Show the plot
plt.show()




# import matplotlib.pyplot as plt
# import matplotlib.colors as plt_color
# fig = plt.figure() #调用figure创建一个绘图对象
# ax = fig.add_subplot(111)
# cax = ax.matshow(initlal-initlal2, cmap='bwr',vmin=-0.3, vmax=0.3)  #绘制热力图，从-1到1
# #cax = ax.matshow(opt-proposed, cmap='bwr',vmin=-0.3, vmax=0.3)  #绘制热力图，从-1到1

# #ax.axis('off')
# #cax = ax.matshow(correlations, cmap='Blues',vmin=np.min(k2), vmax=np.max(k2))  #绘制热力图，从-1到1
# fig.colorbar(cax)  #将matshow生成热力图设置为颜色渐变条
# #ticks = np.arange(0,9,1) #生成0-9，步长为1
# #ax.set_xticks(ticks)  #生成刻度
# #ax.set_yticks(ticks)
# #ax.set_xticklabels(names) #生成x轴标签
# #ax.set_yticklabels(names)
# #plt.imshow(I, cmap='gray');
# plt.show()


