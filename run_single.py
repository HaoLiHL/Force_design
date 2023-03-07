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
#dataset=np.load('./dataset/aso_hao.npz')
dataset=np.load('./dataset/Salicylic_acid.npz.npz')  
#dataset=np.load('H2CO_mu.npz')
#dataset=np.load('new_glucose.npz')

AFF_train = AFF_E_inv.AFFTrain()

print('---------uracil-----------200-1000-----------')
n_train=400
#n_train=np.array([200,400,600,800,1000])
#n_train=np.array([100])

print(' The N_train is '+repr(n_train)+'--------------------')


task=np.load('saved_model/task_sali.npy',allow_pickle=True).item()
trained_model = np.load('saved_model/trained_model_sali.npy',allow_pickle=True).item()

# task=np.load('saved_model/task_asp.npy',allow_pickle=True).item()
# trained_model = np.load('saved_model/trained_model_asp.npy',allow_pickle=True).item()
#E_target=max(task1['E_train'])[0]+100
E_target = -16000
print("max energy is "+str(max(task['E_train'])[0])+'min energy is '+str(min(task['E_train'])[0]))
print('target is',E_target)
   
#initial = 200
#print('start from',task["E_train"][initial])

    
#Record=AFF_train.inverseE_new( task,trained_model,E_target,ind_initial=initial,tol_MAE=0.01,lr=2e-3,c=0.1,num_step = 3,random_val = 1e-2)
    
#Record=gdml_train.inverseE( task1,trained_model,E_target,ind_initial=initial,tol_MAE=0.5,lr=1e-5,c=1,num_step = 10)

#np.save('saved_model/Record_1128.npy', Record) 
#Record=np.load('saved_model/Record_1128.npy',allow_pickle=True).item()
# R_target1 = np.array([2.1893727664,    0.6339981721,   -0.1655362427,
#                  2.2867935017,    2.0053394372,   -0.0183626472  ,
#                  1.1606808160,    2.7420580054,    0.3209182679,
#                  -0.0561840390,    2.1079466147,    0.5077285244,
#                  -0.1465765448,    0.7383371993,    0.3559978426,
#          0.9727788099,   -0.0232568592,    0.0200145274,
#        3.0511501862,    0.0554424198 ,  -0.4221000908,
#        3.2297915980,    2.4928781759 ,  -0.1654694953,
#        1.2275508612,    3.8059147508,    0.4382171445,
#      -0.9370660007,    2.6549467358  ,  0.7730723219,
#         -1.3723959273,    0.1503411010 ,   0.6303296086,
#        -2.1909122610,   -0.3742506040 ,  -0.3424522747,
#        -2.0111294426,   -0.1513964539  , -1.5125659184,
#         -3.2752819291,   -1.1885971183 ,   0.2646249402,
#        -2.8357287799,   -2.0953144276,    0.6590821842,
#        -4.0098866531,   -1.4354223073,   -0.4843391571,
#         -3.7366576316,   -0.6531109719,    1.0822540843,
#          0.8963405566,   -1.4857017018,   -0.1309494436,
#          2.0951516266,   -2.0545678390,   -0.3901732098,
#        -0.1016731800,   -2.1745531342,   -0.0405607386,
#        2.0430682970 ,  -3.0013832069 ,  -0.4950806971
#     ]).reshape(1,21,3)

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

# R_target = np.array([-6.6875817508,	6.9175307365,	-1.3227916831,
# -6.6573532277,	8.1922735181,	-0.7453349783,
# -5.5532784481,	8.5806682903,	0.0367116703,
# -4.4809645688,	7.695843713,	0.224044618,
# -4.5131450959,	6.4201709308,	-0.3572496673,
# -5.622940808,	6.0017558949,	-1.1370652845,
# -7.5386423162,	6.5992907049,	-1.9261426888,
# -7.4919297868,	8.8799946083,	-0.9003148223,
# -5.5242135748,	9.572477407,	0.4959583087,
# -3.6074212296,	7.9728918132,	0.8172434107,
# -3.4316385698,	5.5631795834,	-0.0610758588,
# -2.5776576418,	5.1106752128,	-1.1195903882,
# -2.5872008333,	5.6204152279,	-2.2422920771,
# -1.7326483065,	3.9809127912,	-0.6184163161,
# -0.817296816,	3.8980087824,	-1.2205405504,
# -2.3217831712,	3.0532928424,	-0.726579536,
# -1.4886031493,	4.1078226584,	0.4462393581,
# -5.7062830828,	4.6434264174,	-1.711113675,
# -6.8797724988,	4.448823306,	-2.4419602857,
# -4.8757541469,	3.7149303727,	-1.5912215439,
# -6.8577189769,	3.5235251884,	-2.7984240103]).reshape(1,21,3)
#R_target = task['R_train'][0,:,:].reshape(1,12,3)
# cur_predict = AFF_train.predict(task,R_target,trained_model)
# print(cur_predict)

AFF_E_inv.compile_scirpts_for_physics_based_calculation_IO(task['R_train'])
# 
# atomic number shall be defined in the beginning of the program once as well
atomic_number = dataset['z']

computational_method = ['PBE', 'PBE', '6-31G']
new_E, new_F = AFF_E_inv.run_physics_baed_calculation(data_proposed[None], atomic_number, computational_method)


ev_to_kcal = 1
#ev_to_kcal = 23.060541945329334

#print(np.array(new_F).shape)
#print(new_E.shape)
#cost = np.sum(np.abs(np.concatenate(new_E)))

print("current real energy is ",new_E[0]*ev_to_kcal)
print('new_E,new_F ',new_F)
print( "real:", task['E_train'][0])

AFF_E_inv.compile_scirpts_for_physics_based_calculation_IO(task['R_train'])
# 
# atomic number shall be defined in the beginning of the program once as well
atomic_number = dataset['z']

computational_method = ['PBE', 'PBE', '6-31G']
new_E, new_F = AFF_E_inv.run_physics_baed_calculation(data[None], atomic_number, computational_method)


ev_to_kcal = 1
#ev_to_kcal = 23.060541945329334

#print(np.array(new_F).shape)
#print(new_E.shape)
#cost = np.sum(np.abs(np.concatenate(new_E)))

print("current real energy is ",new_E[0]*ev_to_kcal)
print('new_E,new_F ',new_F)
print( "real:", task['E_train'][0])
