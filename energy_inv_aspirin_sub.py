#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 13:54:18 2023

@author: lihao
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:44:24 2022

@author: lihao
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 13:21:45 2022

@author: lihao
"""
import numpy as np
from utils import AFF_E_inv

# #dataset=np.load('benzene_old_dft.npz')
# #dataset=np.load('uracil_dft.npz')
#dataset=np.load('./dataset/aso_hao.npz')
# #dataset=np.load('H2CO_mu.npz')
# #dataset=np.load('new_glucose.npz')

# sub_index = []
# for i in range(dataset["E"].shape[0]):
#     if dataset["E"][i]>-17625.9:
#         sub_index.append(i)
# len(sub_index)

# # the energy is not less than -17625.9
# np.savez_compressed('./dataset/aso_hao_sub.npz', E=dataset["E"][sub_index], F=dataset["F"][sub_index,:,:], R=dataset["R"][sub_index,:,:], z=dataset["z"],name =dataset['name'],
#                     theory =dataset['theory'],md5 = dataset['theory'],type =dataset['type']  )
        

dataset_sub=np.load('./dataset/aso_hao_sub.npz')    
#dataset_sub=np.load('./dataset/aso_hao.npz')        

AFF_train = AFF_E_inv.AFFTrain()

print('---------aspirin sub energy -----------200----------')
n_train=200
#n_train=np.array([200,400,600,800,1000])
#n_train=np.array([100])

print(' The N_train is '+repr(n_train)+'--------------------')
#task=np.save('task_test.npy', task)
#task_test{}.npy
# task1=AFF_train.create_task(dataset_sub,n_train,dataset_sub,100,100,10,1e-5)
# candid_range = np.exp(np.arange(-5,5,1))
# # # # #candid_range = np.arange(0.1,0.7,0.1)
# # # # #candid_range = np.arange(1,100,10)
# trained_model = AFF_train.train(task1,candid_range)

# np.save('./dataset/task_asp_sub.npy', task1) 
# np.save('./dataset/trained_model_asp_sub.npy', trained_model) 


task=np.load('dataset/task_asp_sub.npy',allow_pickle=True).item()
trained_model = np.load('dataset/trained_model_asp_sub.npy',allow_pickle=True).item()
#E_target=max(task1['E_train'])[0]+100
E_target = -17630
print("max energy is "+str(max(task['E_train'])[0])+'min energy is '+str(min(task['E_train'])[0]))
print('target is',E_target)
   
initial = 100
print('start from',task["E_train"][initial])

    
Record=AFF_train.inverseE_new( task,trained_model,E_target,ind_initial=initial,tol_MAE=1e-5,lr=1e-3,c=0.01,num_step = 5,random_val =1e-2)
    
#Record=AFF_train.inverseE( task,trained_model,E_target,ind_initial=initial,tol_MAE=0.5,lr=1e-25,c=1,num_step = 10)

#np.save('saved_model/Record_1128.npy', Record) 
#Record=np.load('saved_model/Record_1128.npy',allow_pickle=True).item()
R_target = Record['R_best']
E_var_rec =Record['E_var_rec']
E_best =Record['E_best']
E_predict_rec =Record['E_predict_rec']
loss_rc = Record['loss_rec']




AFF_E_inv.compile_scirpts_for_physics_based_calculation_IO(task['R_train'])
# 
# atomic number shall be defined in the beginning of the program once as well
atomic_number = dataset_sub['z']

computational_method = ['PBE', 'PBE', '6-31G']
new_E, new_F = AFF_E_inv.run_physics_baed_calculation(R_target, atomic_number, computational_method)


ev_to_kcal = 1
#print(np.array(new_F).shape)
#print(new_E.shape)
#cost = np.sum(np.abs(np.concatenate(new_E)))

print("current real energy is ",new_E[0]*ev_to_kcal)
print('new_E,new_F ',new_F)

n_loop = 0
n_atom = task['R_train'].shape[1]
Real_E_record = [task["E_train"][initial][0],new_E[0]*ev_to_kcal]
Predict_E_record = [task["E_train"][initial][0],E_best[0]]
Real_loss_record = []
while n_loop<20:
    
    n_loop += 1
    print('The '+repr(n_loop)+'-th loop \n')
    
    n_train = task['R_train'].shape[0]
    task['R_train'] = np.append(task['R_train'],R_target).reshape(n_train+1,n_atom,-1)
    
    task['F_train'] = np.append(task['F_train'],new_F).reshape(n_train+1,n_atom,-1)
    
    task['E_train'] = np.append(task['E_train'],np.array(new_E[0]*ev_to_kcal)).reshape(-1,1)
    
    #AFF_train=AFF.AFFTrain()
    #candid_range = np.exp(np.arange(-5,5,1))
    candid_range = np.exp(np.arange(-3,5,1))
    #task['lam'] = 1e-10
    trained_model = AFF_train.train(task,sig_candid_F = candid_range)
    
    
    initial=n_train
    #Record=AFF_train.inverseE_new( task,trained_model,E_target,ind_initial=initial,tol_MAE=0.01,lr=1e-3,c=0.01,num_step = 15)
       
    Record=AFF_train.inverseE_new( task,trained_model,E_target,ind_initial=initial,tol_MAE=1e-5,lr=1e-3,c=0.1,num_step = 30,random_val = 1e-3)
     
    R_target = Record['R_best']
    E_var_rec =Record['E_var_rec']
    E_best =Record['E_best']
    E_predict_rec =Record['E_predict_rec']
    loss_rc = Record['loss_rec']
    R_design = Record['R_design']
   
    
    print('another one \n')
    print(R_design.shape)
    if len(R_design)==0:
        
        print('Warning! Cannot further reduce the Loss')
        break
    
    AFF_E_inv.compile_scirpts_for_physics_based_calculation_IO(task['R_train'])
    # 
    # atomic number shall be defined in the beginning of the program once as well
    atomic_number = dataset_sub['z']
   
    computational_method = ['PBE', 'PBE', '6-31G']
    new_E, new_F = AFF_E_inv.run_physics_baed_calculation(R_target, atomic_number, computational_method)
    #cost = np.sum(np.abs(np.concatenate(new_F)-np.concatenate(F_target)))
    #cost = np.abs(E_target-new_E)
    
    Real_E_record.append(new_E[0]*ev_to_kcal)
    Predict_E_record.append(E_best[0])
    #Real_loss_record.append()
    if new_E[0]*ev_to_kcal == 0:
        print("simulation fail, stop!")
        break
    print("current predicted energy is ",E_best)
    print("current real energy is ",new_E[0]*ev_to_kcal)
    print('new_E,new_F ',new_F)

    #print('new_E,new_F ',new_F)
    
print('-------finished-------- \n')
print('REAL E Record', Real_E_record) 
#np.save('Real_E_record_asp.npy', Real_E_record) 


print('Predict E Record', Predict_E_record) 
#np.save('Predict_E_record_asp.npy', Predict_E_record) 

print('save the proposed R')
#np.save('proposed_R_asp.npy',  task) 
