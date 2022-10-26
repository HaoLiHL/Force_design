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

n_train=100

#create the task file contains the training, validation and testing dataset 
task=AFF_train.create_task(train_dataset=dataset,
                             n_train = n_train ,
                             valid_dataset=dataset,
                             n_valid=50,
                             n_test=50,
                             lam = 1e-15)
F_target=np.zeros((task['R_train'].shape[1],3)).reshape(-1)
#np.min(np.sum(np.abs(task['F_train']-F_target),1))
#F_target = task["F_train"][4,:,:].reshape(-1)

#F_target[0]=0

initial=1

print('another one \n')
R_design1 = np.array([[ 0.76684281,  0.23675346,  1.37564978],
       [-0.49601898, -0.13014188,  1.46375798],
       [-1.94452951, -0.42374924,  0.69041259],
       [-1.17257929, -0.33838961, -0.75227569],
       [ 0.37707832,  0.08158707, -0.8606521 ],
       [ 1.31743352,  0.35782701,  0.07219055],
       [-1.71771875, -0.53216988, -1.51332436],
       [ 2.44526085,  0.70815321, -0.14216755],
       [-0.91013728, -0.22467986,  2.99990995],
       [-1.95816742, -0.40055395,  0.54685045],
       [ 0.48446565,  0.23261169, -2.22761349],
       [ 1.71967938,  0.42967512,  2.37536615]])[None]

AFF_test.compile_scirpts_for_physics_based_calculation_IO(R_design1)
# 
# atomic number shall be defined in the beginning of the program once as well
atomic_number = dataset['z']
# 
# for-loop
# suggested computational method for H2CO for better reproduction of literature data
# computational_method = ['mp2', 'aug-cc-pVTZ']
# suggested computational method for uracil
computational_method = ['PBE', 'PBE', '6-31G']
new_E, new_F = AFF_test.run_physics_baed_calculation(R_design1, atomic_number, computational_method)

# computational_method = ['PBE', '6-31G']
# new_E, new_F = AFF.run_physics_baed_calculation(R_val_atom_last[None], atomic_number, computational_method)
print(np.array(new_F).shape)
print(new_F.shape)
cost = np.sum(np.abs(np.concatenate(new_F)))

print("current real cost is ",cost)
print('new_E,new_F ',new_F)


n_loop = 0
print('-------finished-------- \n')

print('show be 11254.82272824458')

example_text1
                              