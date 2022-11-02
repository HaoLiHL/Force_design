#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 13:21:45 2022

@author: lihao
"""
import numpy as np
from utils import AFF_E_inv

#dataset=np.load('benzene_old_dft.npz')
#dataset=np.load('uracil_dft.npz')
dataset=np.load('uracil_dft_mu.npz')
#dataset=np.load('H2CO_mu.npz')
#dataset=np.load('new_glucose.npz')

AFF_train = AFF_E_inv.AFFTrain()

print('---------uracil-----------200-1000-----------')
n_train=400
#n_train=np.array([200,400,600,800,1000])
#n_train=np.array([100])

print(' The N_train is '+repr(n_train)+'--------------------')
#task=np.save('task_test.npy', task)
#task_test{}.npy
# task1=gdml_train.create_task(dataset,n_train[i],dataset,400,500,10,1e-15)
# candid_range = np.exp(np.arange(-5,3,1))
# #candid_range = np.arange(0.1,0.7,0.1)
# #candid_range = np.arange(1,100,10)
# trained_model = gdml_train.train(task1,candid_range)

# np.save('task11.npy', task1) 
# np.save('trained_model1.npy', trained_model) 

task=np.load('saved_model/task11.npy',allow_pickle=True).item()
trained_model = np.load('saved_model/trained_model1.npy',allow_pickle=True).item()

#E_target=max(task1['E_train'])[0]+100
E_target = -259000
print("max energy is "+str(max(task['E_train'])[0])+'min energy is '+str(min(task['E_train'])[0]))
print('target is',E_target)
   
initial = 1
print('start from',task["E_train"][initial])

    
Record=AFF_train.inverseE( task,trained_model,E_target,ind_initial=initial,tol_MAE=0.5,lr=1e-4,c=0)
    
R_target = Record['R_last']
E_var_rec =Record['E_var_rec']
E_predict_rec =Record['E_predict_rec']

AFF_E_inv.compile_scirpts_for_physics_based_calculation_IO(task['R_train'])
# 
# atomic number shall be defined in the beginning of the program once as well
atomic_number = dataset['z']

computational_method = ['PBE', 'PBE', '6-31G']
new_E, new_F = AFF_E_inv.run_physics_baed_calculation(R_target[None], atomic_number, computational_method)

#print(np.array(new_F).shape)
#print(new_E.shape)
#cost = np.sum(np.abs(np.concatenate(new_E)))

print("current real energy is ",new_E[0]*23.06)
print('new_E,new_F ',new_F)

n_loop = 0

Real_E_record = [task["E_train"][initial][0],new_E[0]*23.06]

while n_loop<20:
    
    n_loop += 1
    print('The '+repr(n_loop)+'-th loop \n')
    
    n_train = task['R_train'].shape[0]
    task['R_train'] = np.append(task['R_train'],R_target[None]).reshape(n_train+1,12,-1)
    
    task['F_train'] = np.append(task['F_train'],new_F).reshape(n_train+1,12,-1)
    
    task['E_train'] = np.append(task['E_train'],np.array(new_E[0]*23.06)).reshape(-1,1)
    
    #AFF_train=AFF.AFFTrain()
    candid_range = np.exp(np.arange(-5,4,1))
    trained_model = AFF_train.train(task,sig_candid_F = candid_range)
    
    
    initial=n_train
    
    Record=AFF_train.inverseE( task,trained_model,E_target,ind_initial=initial,tol_MAE=0.5,lr=1e-4,c=0)
        
    R_target = Record['R_last']
    E_var_rec =Record['E_var_rec']
    E_predict_rec =Record['E_predict_rec']
    R_design = Record['R_design']
   
    
    print('another one \n')
    print(R_design.shape)
    if len(R_design)==0:
        
        print('Warning! Cannot further reduce the Loss')
        break
    
    AFF_E_inv.compile_scirpts_for_physics_based_calculation_IO(task['R_train'])
    # 
    # atomic number shall be defined in the beginning of the program once as well
    atomic_number = dataset['z']
   
    computational_method = ['PBE', 'PBE', '6-31G']
    new_E, new_F = AFF_E_inv.run_physics_baed_calculation(R_target[None], atomic_number, computational_method)
    #cost = np.sum(np.abs(np.concatenate(new_F)-np.concatenate(F_target)))
    #cost = np.abs(E_target-new_E)
    
    Real_E_record.append(new_E[0]*23.06)
    
    if new_E[0]*23.06 == 0:
        print("simulation fail, stop!")
        break

    print("current real energy is ",new_E[0]*23.06)
    print('new_E,new_F ',new_F)

    #print('new_E,new_F ',new_F)
    
print('-------finished-------- \n')
print('REAL E Record', Real_E_record) 
np.save('Real_E_record.npy', Real_E_record) 
    # Record1=gdml_train.inverseE( task1,trained_model,E_target,ind_initial=initial,tol_MAE=0.5,lr=1e-7,c=1e7)
    
    # R_target1 = Record1['R_last']
    # E_var_rec1 =Record1['E_var_rec']
    # E_predict_rec1 =Record1['E_predict_rec']
    #{'R_last':R_last,'E_var_rec':E_var_rec,'E_predict_rec':E_predict_rec}
    # n_sam=20
    # R_target=np.empty((n_sam,12,3))
    # for l in range(n_sam):
    #     print(' 259950: it is the  '+repr(l)+'-th sample-------------------')
    #     R_target[l,:,:]=gdml_train.inverseE( task1,sig_opt,theta_hat,alphas_opt,E_target,ind_initial=l,tol_MAE=0.03,lr=1e-12)
    
    #E_target1=-259950
    
    # R_target1=np.empty((n_sam,12,3))
    # for l in range(n_sam):
    #     print(' 259950: it is the  '+repr(l)+'-th sample-------------------')
    #     R_target1[l,:,:]=gdml_train.inverseE( task1,sig_opt,theta_hat,alphas_opt,E_target1,ind_initial=l,tol_MAE=0.03)
    
        
    #R_target=gdml_train.inverseE( task1,sig_opt,theta_hat,alphas_opt,E_target,ind_initial=0,tol_MAE=0.03)
    
#np.save('R_target_259920.npy', R_target) 
#np.save('R_target_259950.npy', R_target1) 

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots((2))

# #plt.plot(np.abs(E_predict_rec))
# ax[0].plot(np.arange(len(E_var_rec)),np.array(E_var_rec).reshape(-1), color = 'r')
# ax[0].set_title('var')

# ax[1].plot(np.arange(len(E_predict_rec)),np.array(np.abs(E_predict_rec)).reshape(-1))
# ax[1].set_title('mean')
# fig.show()


# fig, ax = plt.subplots((2))

# #plt.plot(np.abs(E_predict_rec))
# ax[0].plot(np.arange(len(E_var_rec)),np.array(E_var_rec).reshape(-1), color = 'r')
# ax[0].plot(np.arange(len(E_var_rec1)),np.array(E_var_rec1).reshape(-1), color = 'b')
# ax[0].set_title('var')

# ax[1].plot(np.arange(len(E_predict_rec)),np.array(np.abs(E_predict_rec)).reshape(-1), color = 'r')
# ax[1].plot(np.arange(len(E_predict_rec1)),np.array(np.abs(E_predict_rec1)).reshape(-1), color = 'b')
# ax[1].set_title('mean')
# plt.show()



# ---------------plot the R_target------------------------------
#R=task1['R_test']
#R_target=R[slice(1),:,:]
# import plotly.graph_objects as go
# import numpy as np
# #import plotly.graph_objects as go
# import plotly.io as pio
# #pio.renderers.default = 'svg'  # change it back to spyder
# pio.renderers.default = 'browser'

# # Helix equation
# #t = np.linspace(0, 10, 50)
# R_target1 = R_target[None]
# x, y, z = R_target1[0,:,0],R_target1[0,:,1],R_target1[0,:,2]

# fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
#                                   mode='markers')])
# fig.show()
