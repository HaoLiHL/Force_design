#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:54:22 2022

@author: lihao
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:21:08 2022

@author: hao li
"""




import numpy as np
from utils import AFF_test


# load the dataset contains the geometry information and the force-fields

#dataset=np.load('./dataset/aspirin_ccsd-train.npz')
dataset=np.load('uracil_dft_mu.npz')
#dataset=np.load('H2CO_mu.npz')
AFF_train=AFF_test.AFFTrain()

c

#create the task file contains the training, validation and testing dataset 
# task=AFF_train.create_task(train_dataset=dataset, 
#                             n_train = n_train ,
#                             valid_dataset=dataset,
#                             n_valid=50,
#                             n_test=50,
#                             lam = 1e-15)
#np.save('uracil_task',task)
task = np.load('uracil_task.npy',allow_pickle=True).item()
trained_model = np.load('uracil_trained_model.npy',allow_pickle = True).item()
# start training the model based on the training dataset
#trained_model = AFF_train.train(task,sig_candid_F = np.arange(10,100,30))
#np.save('uracil_trained_model',trained_model)
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

initial=50
#n_sam=11
#R_target1=np.empty((n_sam,12,3))
#F_predict=np.empty((n_sam,12,3))
#cost=np.empty((n_sam,1))

#np.array(R_design),R_val_atom_last,F_hat,record,cost_SAE
# R_design,R_val_atom_last,F_hat,record,cost_SAE = AFF_train.inverseF_test(task,
#                                                                     trained_model,                                                                
#                                                                     initial,
#                                                                     F_target,
#                                                                     lr=1e-10)

# R_design,R_val_atom_last,F_hat,record,cost_SAE = AFF_train.inverseF_test(task,
#                                                                     trained_model,                                                                
#                                                                     initial,
#                                                                     F_target,
#                                                                      lr=1e-15)

R_design,R_val_atom_last,F_hat,record,cost_SAE = AFF_train.inverseF(task,
                                                                    trained_model,                                                                
                                                                    initial,
                                                                    F_target,
                                                                     lr=1e-10)


# print('another one \n')

AFF_test.compile_scirpts_for_physics_based_calculation_IO(R_design)
# 
# atomic number shall be defined in the beginning of the program once as well
atomic_number = dataset['z']
# 
# for-loop
# suggested computational method for H2CO for better reproduction of literature data
# computational_method = ['mp2', 'aug-cc-pVTZ']
# suggested computational method for uracil

computational_method = ['PBE', 'PBE', '6-31G']
new_E, new_F = AFF_test.run_physics_baed_calculation(R_val_atom_last[None], atomic_number, computational_method)

# computational_method = ['PBE', '6-31G']
# new_E, new_F = AFF.run_physics_baed_calculation(R_val_atom_last[None], atomic_number, computational_method)
print(np.array(new_F).shape)
print(new_F.shape)
cost = np.sum(np.abs(np.concatenate(new_F)))

print("current real cost is ",cost)
print('new_E,new_F ',new_F)

n_loop = 0

while n_loop<20:
    
    n_loop += 1
    print('The '+repr(n_loop)+'-th loop \n')
    
    n_train = task['R_train'].shape[0]
    task['R_train'] = np.append(task['R_train'],R_val_atom_last[None]).reshape(n_train+1,12,-1)
    
    task['F_train'] = np.append(task['F_train'],new_F).reshape(n_train+1,12,-1)
    
    task['E_train'] = np.append(task['E_train'],new_E).reshape(-1,1)
    
    #AFF_train=AFF.AFFTrain()
    trained_model = AFF_train.train(task,sig_candid_F = np.arange(10,100,30))
    
    
    initial=n_train
    
    R_design,R_val_atom_last,F_hat,record,cost_SAE = AFF_train.inverseF_test(task,
                                                                        trained_model,                                                                
                                                                        initial,
                                                                        F_target,
                                                                        lr=1e-10)
    
    print('another one \n')
    print(R_design.shape)
    if len(R_design)==0:
        
        print('Warning! Cannot further reduce the Loss')
        break
    
    AFF_test.compile_scirpts_for_physics_based_calculation_IO(R_design)
    # 
    # atomic number shall be defined in the beginning of the program once as well
    atomic_number = dataset['z']
    # 
    # for-loop
    # suggested computational method for H2CO for better reproduction of literature data
    # computational_method = ['mp2', 'aug-cc-pVTZ']
    # suggested computational method for uracil
    computational_method = ['PBE', 'PBE', '6-31G']
    new_E, new_F = AFF_test.run_physics_baed_calculation(R_val_atom_last[None], atomic_number, computational_method)
    #cost = np.sum(np.abs(np.concatenate(new_F)-np.concatenate(F_target)))
    cost = np.sum(np.abs(np.concatenate(new_F)))
    print("current real cost is ",cost)
    #print('new_E,new_F ',new_F)
    
print('-------finished-------- \n')






# # #task = np.load('uracil_task.npy',allow_pickle=True).item()
# # #trained_model = np.load('uracil_trained_model.npy',allow_pickle = True).item()
# # #trained_model = AFF_train.train(task,sig_candid_F = np.arange(10,100,30))
# # # R_design,R_val_atom_last,F_hat,record,cost_SAE = AFF_train.inverseF(task,
# # #                                                                     trained_model,                                                                
# # #                                                                     initial,
# # #                                                                     F_target,
# # #                                                                     lr=1e-11)
