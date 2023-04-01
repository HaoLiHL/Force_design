#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 23:56:16 2023

@author: lihao
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 13:21:45 2022

@author: lihao
"""
import numpy as np
from utils import tensor_aff

#dataset=np.load('benzene_old_dft.npz')
#dataset=np.load('uracil_dft.npz')
#dataset=np.load('./dataset/h2c0_hao.npz')
dataset=np.load('./dataset/h2_long.npz')
#dataset=np.load('H2CO_mu.npz')
#dataset=np.load('new_glucose.npz')

AFF_train = tensor_aff.GDMLTrain()

print('---------uracil-----------200-1000-----------')
n_train=200
#n_train=np.array([200,400,600,800,1000])
#n_train=np.array([100])

print(' The N_train is '+repr(n_train)+'--------------------')


#candid_range = np.exp(np.arange(1,10,2))
# candid_range = np.arange(1,70,10)
# task=AFF_train.create_task(dataset,n_train,dataset,100,50,100,1e-12)
# # #candid_range = np.exp(np.arange(-2,5,1))
# trained_model= AFF_train.train(task,candid_range,np.arange(0.1,1,0.1))
# np.save('saved_model/task_h2_long.npy', task) 
# np.save('saved_model/trained_model_h2_long.npy', trained_model) 


task=np.load('saved_model/task_h2_long.npy',allow_pickle=True).item()
trained_model=np.load('saved_model/trained_model_h2_long.npy',allow_pickle=True).item()


# E_target = -17630
print("max energy is "+str(max(task['E_train'])[0])+'min energy is '+str(min(task['E_train'])[0]))
#print('target is',E_target)
   
initial = 150
print('start from',task["E_train"][initial])


# this one freq = 2
initial_position =  np.array([[ 0.62296138, -0.51916103, -0.01964953],
       [-0.49433722,  0.08627029, -0.79764296],
       [ 1.6632841 , -0.88922841, -0.02380747],
       [-0.20417143, -0.72422711,  0.3661379 ]])
# np.array([[0.6493005159, 0.3335238106, -0.0001173225],
#                 [-0.6516737070, -0.0006357680, 0.0001145444],
#                 [1.2202008073, -0.9870460045, 0.0009573408],
#                 [0.0973857531, -1.0090107149, -0.0011697607]])



R_proposed_tensor,F_predict = AFF_train.inverse(task,trained_model,initial_position=initial_position, c = 1e-5, n_iter = 30, step_size= 1e-3)   

F_predict_pro = F_predict#.cpu().detach().numpy()
R_target = R_proposed_tensor.cpu().detach().numpy()

tensor_aff.compile_scirpts_for_physics_based_calculation_IO(task['R_train'])
# 
# atomic number shall be defined in the beginning of the program once as well
atomic_number = dataset['z']

computational_method = ['PBE', 'PBE', '6-31G']
new_E, new_F = tensor_aff.run_physics_baed_calculation(R_target[None], atomic_number, computational_method)


ev_to_kcal = 1
#print(np.array(new_F).shape)
#print(new_E.shape)
#cost = np.sum(np.abs(np.concatenate(new_E)))

print("current real energy is ",new_E[0]*ev_to_kcal)
print('new_E,new_F ',new_F)

n_loop = 0
n_atom = task['R_train'].shape[1]
Real_E_record = [task["E_train"][initial][0],new_E[0]*ev_to_kcal]
Real_F_loss_record = [np.linalg.norm(task["F_train"][initial,:,:])**2,np.linalg.norm(new_F)**2]
Real_loss_record = []
while n_loop<5:
    
    n_loop += 1
    print('The '+repr(n_loop)+'-th loop \n')
    
    n_train = task['R_train'].shape[0]
    task['R_train'] = np.append(task['R_train'],R_target[None]).reshape(n_train+1,n_atom,-1)
    
    task['F_train'] = np.append(task['F_train'],new_F).reshape(n_train+1,n_atom,-1)
    
    task['E_train'] = np.append(task['E_train'],np.array(new_E[0]*ev_to_kcal)).reshape(-1,1)
    
    #AFF_train=AFF.AFFTrain()
    #candid_range = np.exp(np.arange(-5,5,1))
    #candid_range = np.exp(np.arange(-2,2,1))
    candid_range =  np.arange(10,21,10)
    #task['lam'] = 1e-10
    trained_model = AFF_train.train(task,candid_range,np.arange(0.1,1,0.1))
    #AFF_train.train(task,sig_candid_F = candid_range)
    
    if np.linalg.norm(new_F)<=np.linalg.norm(task['F_train'][initial,:,:]):
        initial_position=R_target
    print('initial is ', initial)
    #initial=n_train
    #Record=AFF_train.inverseE_new( task,trained_model,E_target,ind_initial=initial,tol_MAE=0.01,lr=1e-3,c=0.01,num_step = 15)
       
    R_design_tensor,F_predict=AFF_train.inverse(task,trained_model,initial_position=initial_position, c = 1e-5, n_iter = 100,random_noise = 1e-4,step_size=1e-3)   

    #AFF_train.inverseE_new( task,trained_model,E_target,ind_initial=initial,tol_MAE=0.01,lr=1e-1,c=10,num_step = 30,random_val = 1e-2)
    #R_proposed_tensor,F_predict = AFF_train.inverse(task,trained_model,initial=initial, c = 1e-5, n_iter = 200)   

      #F_predict_pro = F_predict#.cpu().detach().numpy()
    R_target = R_design_tensor.cpu().detach().numpy()
    #F_predict_loss = np.linalg.norm(F_predict)
   
    print('THE Proposed Position',R_target)
    
    print('another one \n')
    print(R_target.shape)
    if len(R_target)==0:
        
        print('Warning! Cannot further reduce the Loss')
        break
    
    tensor_aff.compile_scirpts_for_physics_based_calculation_IO(task['R_train'])
    # 
    # atomic number shall be defined in the beginning of the program once as well
    atomic_number = dataset['z']
   
    computational_method = ['PBE', 'PBE', '6-31G']
    new_E, new_F = tensor_aff.run_physics_baed_calculation(R_target[None], atomic_number, computational_method)
    #cost = np.sum(np.abs(np.concatenate(new_F)-np.concatenate(F_target)))
    #cost = np.abs(E_target-new_E)
    
    
    Real_E_record.append(new_E[0]*ev_to_kcal)
    Real_F_loss_record.append(np.linalg.norm(new_F)**2)
    #Real_loss_record.append()
    if new_E[0]*ev_to_kcal == 0:
        print("simulation fail, stop!")
        break
    print("current real energy is ",new_E,'\n')
    print("current real F loss is ",np.linalg.norm(new_F)**2,'\n')
    print('current real force ',new_F,'\n')
    print('REAL F loss Record', Real_F_loss_record) 

    #print('new_E,new_F ',new_F)
n_train = task['R_train'].shape[0]
task['R_train'] = np.append(task['R_train'],R_target[None]).reshape(n_train+1,n_atom,-1)
Proposed_h2co = {'R_design':R_target,'Real_F':new_F,'R_train':task['R_train']}    
print('-------finished-------- \n')
print('REAL E Record', Real_E_record) 
print('Final Proposed Position',R_target)
print('REAL F loss Record', Real_F_loss_record) 
np.save('saved_model/Design_E_record_h2_long.npy', Proposed_h2co) 





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
