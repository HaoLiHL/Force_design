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

#dataset=np.load('benzene_old_dft.npz')
#dataset=np.load('uracil_dft.npz')
dataset=np.load('./dataset/aso_hao.npz')
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


task=np.load('saved_model/task_asp.npy',allow_pickle=True).item()
trained_model = np.load('saved_model/trained_model_asp.npy',allow_pickle=True).item()
#E_target=max(task1['E_train'])[0]+100
E_target = -16000
print("max energy is "+str(max(task['E_train'])[0])+'min energy is '+str(min(task['E_train'])[0]))
print('target is',E_target)
   
initial = 200
print('start from',task["E_train"][initial])

    
Record=AFF_train.inverseE_new( task,trained_model,E_target,ind_initial=initial,tol_MAE=0.01,lr=2e-3,c=0.1,num_step = 3,random_val = 1e-2)
    
#Record=gdml_train.inverseE( task1,trained_model,E_target,ind_initial=initial,tol_MAE=0.5,lr=1e-5,c=1,num_step = 10)

#np.save('saved_model/Record_1128.npy', Record) 
#Record=np.load('saved_model/Record_1128.npy',allow_pickle=True).item()
R_target1 = np.array([2.1893727664,    0.6339981721,   -0.1655362427,
                 2.2867935017,    2.0053394372,   -0.0183626472  ,
                 1.1606808160,    2.7420580054,    0.3209182679,
                 -0.0561840390,    2.1079466147,    0.5077285244,
                 -0.1465765448,    0.7383371993,    0.3559978426,
         0.9727788099,   -0.0232568592,    0.0200145274,
       3.0511501862,    0.0554424198 ,  -0.4221000908,
       3.2297915980,    2.4928781759 ,  -0.1654694953,
       1.2275508612,    3.8059147508,    0.4382171445,
     -0.9370660007,    2.6549467358  ,  0.7730723219,
        -1.3723959273,    0.1503411010 ,   0.6303296086,
       -2.1909122610,   -0.3742506040 ,  -0.3424522747,
       -2.0111294426,   -0.1513964539  , -1.5125659184,
        -3.2752819291,   -1.1885971183 ,   0.2646249402,
       -2.8357287799,   -2.0953144276,    0.6590821842,
       -4.0098866531,   -1.4354223073,   -0.4843391571,
        -3.7366576316,   -0.6531109719,    1.0822540843,
         0.8963405566,   -1.4857017018,   -0.1309494436,
         2.0951516266,   -2.0545678390,   -0.3901732098,
       -0.1016731800,   -2.1745531342,   -0.0405607386,
       2.0430682970 ,  -3.0013832069 ,  -0.4950806971
    ]).reshape(1,21,3)

R_target = np.array([2.2202463525,	0.6110479309,	0.1146075899,
2.3178666508,	1.9875864405,	0.0273553355,
1.1957315376,	2.7374930538,	-0.295499341,
-0.0179264935,	2.1112291485,	-0.5239258395,
-0.1084945039,	0.7364929389,	-0.4312842179,
1.0073784294,	-0.0385503782,	-0.1158405814,
3.0781746007,	0.0221114701,	0.3601111901,
3.2576258653,	2.4698651724,	0.2080517896,
1.2627305796,	3.8053609756,	-0.3670025625,
-0.8972178851,	2.668317655,	-0.7733740841,
-1.3331739358,	0.1626701083,	-0.7388443016,
-2.1612703108,	-0.3897792571,	0.2091845682,
-1.9760797221,	-0.227087396,	1.3884835139,
-3.2637713867,	-1.1508343061,	-0.4331013094,
-4.0229525025,	-1.3802092413,	0.2970711644,
-3.6845815197,	-0.5844831529,	-1.2512000685,
-2.8534813746,	-2.0685421158,	-0.8338649873,
0.9321781067,	-1.5064495786,	-0.0300680896,
2.1293741945,	-2.0836479463,	0.2212930933,
-0.0629815348,	-2.1923526734,	-0.1645777984,
2.0775145591,	-3.0339890298,	0.2876882979]).reshape(1,21,3)



AFF_E_inv.compile_scirpts_for_physics_based_calculation_IO(task['R_train'])
# 
# atomic number shall be defined in the beginning of the program once as well
atomic_number = dataset['z']

computational_method = ['PBE', 'PBE', '6-31G']
new_E, new_F = AFF_E_inv.run_physics_baed_calculation(R_target, atomic_number, computational_method)


ev_to_kcal = 1
#print(np.array(new_F).shape)
#print(new_E.shape)
#cost = np.sum(np.abs(np.concatenate(new_E)))

print("current real energy is ",new_E[0]*ev_to_kcal)
print('new_E,new_F ',new_F)
