#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 22:07:39 2022

@author: lihao
"""

import numpy as np
from utils import AFF_test

# load the dataset contains the geometry information and the force-fields

#dataset=np.load('./dataset/aspirin_ccsd-train.npz')
#dataset=np.load('uracil_dft_mu.npz')
#dataset=np.load('H2CO_mu.npz')
AFF_train=AFF_test.AFFTrain()

n_train=200

# create the task file contains the training, validation and testing dataset 
# task=AFF_train.create_task(train_dataset=dataset, 
#                             n_train = n_train ,
#                             valid_dataset=dataset,
#                             n_valid=100,
#                             n_test=50,
#                             lam = 1e-15)
# np.save('h2c0_task',task)


# trained_model = AFF_train.train(task,sig_candid_F = np.arange(1,100,20))
# np.save('h2c0_trained_model',trained_model)

task = np.load('h2c0_task.npy',allow_pickle=True).item()
trained_model = np.load('h2c0_trained_model.npy',allow_pickle = True).item()

# task = np.load('uracil_task.npy',allow_pickle=True).item()
# trained_model = np.load('uracil_trained_model.npy',allow_pickle = True).item()



# start training the model based on the training dataset
#trained_model = AFF_train.train(task,sig_candid_F = np.arange(10,100,30))
# np.save('h2c0_trained_model',trained_model)

#trained_model = np.load('uracil_trained_model.npy',allow_pickle = True).item()

# predicted the force-field using the trained_model
#prediction=AFF_train.predict(task = task, trained_model = trained_model)

#task['R_train'] = np.append(task['R_train'],task['R_test'][0,:,:]).reshape(101,12,-1)

#task['F_train'] = np.append(task['F_train'],task['F_test'][0,:,:]).reshape(101,12,-1)

#task['E_train'] = np.append(task['E_train'],task['E_test'][0]).reshape(-1,1)

#trained_model = AFF_train.train(task,sig_candid_F = np.arange(10,100,10))

F_target=np.zeros((task['R_train'].shape[1],3)).reshape(-1)
#np.min(np.sum(np.abs(task['F_train']-F_target),1))
#F_target = task["F_train"][4,:,:].reshape(-1)

#F_target[0]=0
min_value = 10000
initial = 0
for i in range(task['R_train'].shape[0]):
    #print(np.sum(np.abs(np.concatenate(F_hat_val_F)-np.concatenate(F_hat_val_target))))
    val = np.sum(np.abs(task['F_train'][i,:,:].reshape(-1)-F_target))
    #print((np.sum(np.abs(task['F_train'][i,:,:].reshape(-1)-F_target))))
    if val < min_value:
        initial = i
        min_value = val
#print(min_value)
initial=17

#R_val = task['R_test'][1,:,:].reshape(36)
#df_dr,F_val = AFF_train.inverseF_debug(task,trained_model,initial,F_target,lr=1e-10,R_val=R_val)

# R_val1 = R_val + np.concatenate([np.array([0.000001]),np.repeat(0,35)])
# df_dr1,F_val2 = AFF_train.inverseF_debug(task,trained_model,initial,F_target,lr=1e-10,R_val=R_val1)

# print((F_val2- F_val)/0.00000001)
#R_design,R_val_atom_last,F_hat,record,cost_SAE = AFF_train.inverseF(task,trained_model,initial,F_target,lr=1e-9)


#*** testing derivative *******
R_val = task['R_test'][1,:,:].reshape(task['R_train'].shape[1]*3)
# #df_dr,F_val = AFF_train.inverseF_debug_22(task,trained_model,initial,F_target,lr=1e-10,R_val=R_val)

df_dr,F_hat = AFF_train.inverseF_debug_22(task,trained_model,initial,F_target,lr=1e-12,R_val = R_val)


d_x = 1e-5
R_val1 = R_val + np.concatenate([np.array([d_x]),np.repeat(0,task['R_train'].shape[1]*3-1)])
df_dr2,F_hat2 = AFF_train.inverseF_debug_22(task,trained_model,initial,F_target,lr=1e-12,R_val = R_val1)

print((F_hat2-F_hat)/d_x)
#*** testing derivative *******-------end


# samkk molecular: 
# select the training set from 30 - 100
# to see if we could find the sample less than 30
#
# 


#print((predicted_var- predicted_var2)/0.00000001)
#print('analynical variance',drl_var)
#df_dr1,F_val2 = AFF_train.inverseF_debug_22(task,trained_model,initial,F_target,lr=1e-10,R_val=R_val1)

# R_val = task['R_test'][1,:,:].reshape(task['R_train'].shape[1]*3)
# #df_dr,F_val = AFF_train.inverseF_debug_22(task,trained_model,initial,F_target,lr=1e-10,R_val=R_val)

# df_dr,F_hat,drl_var,predicted_var = AFF_train.inverseF_debug22(task,trained_model,initial,F_target,lr=1e-12,R_val = R_val)


# R_val1 = R_val + np.concatenate([np.array([0.00000001]),np.repeat(0,task['R_train'].shape[1]*3-1)])
# df_dr2,F_hat2,drl_var2,predicted_var2 = AFF_train.inverseF_debug22(task,trained_model,initial,F_target,lr=1e-12,R_val = R_val1)

# print((predicted_var- predicted_var2)/0.00000001)
# print('analynical variance',drl_var)
# #df_dr1,F_val2 = AFF_train.inverseF_debug_22(task,trained_model,initial,F_target,lr=1e-10,R_val=R_val1)

# print((F_hat-F_hat2)/0.00000001)
# # #print((F_val2[0,0]- F_val[0,0])/0.00000001)


