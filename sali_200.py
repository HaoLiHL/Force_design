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
dataset=np.load('./dataset/sali200_hao.npz')
#dataset=np.load('H2CO_mu.npz')
#dataset=np.load('new_glucose.npz')

AFF_train = tensor_aff.GDMLTrain()

print('---------uracil-----------200-1000-----------')
n_train=200
#n_train=np.array([200,400,600,800,1000])
#n_train=np.array([100])

print(' The N_train is '+repr(n_train)+'--------------------')


# candid_range = np.exp(np.arange(-2,5,1))
# task=AFF_train.create_task(dataset,n_train,dataset,100,50,100,1e-12)
# # #candid_range = np.exp(np.arange(-2,5,1))
# trained_model= AFF_train.train(task,candid_range,np.arange(0.1,1,0.1))
# np.save('saved_model/task_sali20.npy', task) 
# np.save('saved_model/trained_model_sali20.npy', trained_model) 

task=np.load('saved_model/task_sali.npy',allow_pickle=True).item()
trained_model=np.load('saved_model/trained_model_sali.npy',allow_pickle=True).item()
#task=np.load('saved_model/task_sali20.npy',allow_pickle=True).item()
#trained_model=np.load('saved_model/trained_model_sali20.npy',allow_pickle=True).item()


# E_target = -17630
print("max energy is "+str(max(task['E_train'])[0])+'min energy is '+str(min(task['E_train'])[0]))
#print('target is',E_target)
   
initial = 10
initial_position = np.array([[-1.843376345456423, 2.8294932215287116, -0.01622855304226896],
                [-2.544906784610215, 1.5585182784961846, 0.09307981921375645],
                [-1.7794574659569309, 0.3912049696336026, 0.06233784357385165],
                [-0.3481335872889513, 0.2025357079913076, 0.16458879372347054],
                [0.0038257905490962085, 1.520389262637608, -0.17198237234171462],
                [-0.3995710293613946, 2.550976515218652, 0.12582220379395642],
                [1.6758854576297693, 1.6375102983696697, -0.02727739766052799],
                [0.11632659401341171, 3.980990347815039, 0.02081784153651932],
                [1.3938100568195515, 4.2920597317453995, 0.17320720158723465],
                [2.298950987328381, 0.4193423319770672, -0.07363919824452603],
                [3.377872377845014, 0.19338411524454063, -0.05440255047341536],
                [2.612436490449355, 0.336259860646772, 0.023637793301325295],
                [-2.1441253348196567, 3.640971992242205, -0.07275576937647563],
                [-3.6688240006247925, 1.5929112557955432, 0.12030321099092159],
                [-2.240263701129696, -0.5506339077244986, 0.17399340122796053],
                [0.00045691230936797, -0.6822500682045702, 0.08321190183200469]])


#task['R_train'][initial,:,:]
print('start from',task["E_train"][initial])

R_proposed_tensor,F_predict = AFF_train.inverse(task,trained_model,initial_position=initial_position, c = 1e-2, n_iter = 100, step_size= 1e-2)   

F_predict_pro = F_predict#.cpu().detach().numpy()
R_target = R_proposed_tensor.cpu().detach().numpy()

tensor_aff.compile_scirpts_for_physics_based_calculation_IO(task['R_train'])
# 
# atomic number shall be defined in the beginning of the program once as well
atomic_number = dataset['z']

computational_method = ['PBE', 'PBE', '6-31G']
new_E, new_F = tensor_aff.run_physics_baed_calculation(R_target[None], atomic_number, computational_method)
old_F = new_F

ev_to_kcal = 1
#print(np.array(new_F).shape)
#print(new_E.shape)
#cost = np.sum(np.abs(np.concatenate(new_E)))

print("current real energy is ",new_E[0]*ev_to_kcal)
print('new_E,new_F ',new_F)

n_loop = 0
n_atom = task['R_train'].shape[1]
Real_E_record = [task["E_train"][initial][0],new_E[0]*ev_to_kcal]
Real_F_loss_record = [np.linalg.norm(task["F_train"][initial,:,:]),np.linalg.norm(new_F)**2]
Real_loss_record = []
while n_loop<2:
    
    n_loop += 1
    print('The '+repr(n_loop)+'-th loop \n')
    
    n_train = task['R_train'].shape[0]
    task['R_train'] = np.append(task['R_train'],R_target[None]).reshape(n_train+1,n_atom,-1)
    
    task['F_train'] = np.append(task['F_train'],new_F).reshape(n_train+1,n_atom,-1)
    
    task['E_train'] = np.append(task['E_train'],np.array(new_E[0]*ev_to_kcal)).reshape(-1,1)
    
    #AFF_train=AFF.AFFTrain()
    #candid_range = np.exp(np.arange(-5,5,1))
    #candid_range = np.exp(np.arange(-2,2,1))
    candid_range = np.arange(10,20,5)
    #task['lam'] = 1e-10
    trained_model = AFF_train.train(task,candid_range,np.arange(0.1,1,0.1))
    #AFF_train.train(task,sig_candid_F = candid_range)
    
    if np.linalg.norm(new_F)<=np.linalg.norm(old_F):
        initial=n_train
        initial_position = R_target
        old_F = new_F
    print('initial is ', initial)
    #initial=n_train
    #Record=AFF_train.inverseE_new( task,trained_model,E_target,ind_initial=initial,tol_MAE=0.01,lr=1e-3,c=0.01,num_step = 15)
    if n_loop>15:
        R_design_tensor,F_predict=AFF_train.inverse(task,trained_model,initial_position=initial_position, c = 1e-2, n_iter = 100,random_noise = 1e-4,step_size=1e-3)   
    else:
        R_design_tensor,F_predict=AFF_train.inverse(task,trained_model,initial_position=initial_position, c = 1e-2, n_iter = 100,random_noise = 1e-4,step_size=1e-2)   

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
np.save('saved_model/Design_E_record_sali200.npy', Proposed_h2co) 





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
