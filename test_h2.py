
import numpy as np
from utils import AFF_E_inv

#dataset=np.load('benzene_old_dft.npz')
#dataset=np.load('uracil_dft.npz')
#dataset=np.load('./dataset/aso_hao.npz')
dataset=np.load('./dataset/h2_long.npz')
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

data_proposed =  np.array([[0.6493005159, 0.3335238106, -0.0001173225],
                [-0.6516737070, -0.0006357680, 0.0001145444],
                [1.2202008073, -0.9870460045, 0.0009573408],
                [0.0973857531, -1.0090107149, -0.0011697607]])





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
print("real f loss:",np.linalg.norm(new_F)**2)



# import numpy as np
# from utils import tensor_aff

# #dataset=np.load('benzene_old_dft.npz')
# #dataset=np.load('uracil_dft.npz')
# #dataset=np.load('./dataset/h2c0_hao.npz')
# dataset=np.load('./dataset/h2_long.npz')
# #dataset=np.load('H2CO_mu.npz')
# #dataset=np.load('new_glucose.npz')

# AFF_train = tensor_aff.GDMLTrain()

# print('---------uracil-----------200-1000-----------')
# n_train=200
# #n_train=np.array([200,400,600,800,1000])
# #n_train=np.array([100])

# print(' The N_train is '+repr(n_train)+'--------------------')

# task=np.load('saved_model/task_h2_long.npy',allow_pickle=True).item()
# trained_model=np.load('saved_model/trained_model_h2_long.npy',allow_pickle=True).item()



# new1 =  np.array([[0.6493005159, 0.3335238106, -0.0001173225],
#                 [-0.6516737070, -0.0006357680, 0.0001145444],
#                 [1.2202008073, -0.9870460045, 0.0009573408],
#                 [0.0973857531, -1.0090107149, -0.0011697607]]).reshape(1,4,3)

# #new = task['R_test'][0:2,:,:]

# new = np.concatenate((task['R_test'][0,:,:][None], new1), axis=0)


# #AFF_train.predict(task,trained_model,new)
# result = AFF_train.predict(task,trained_model,new)
# print(result)
# #result = AFF_train.test(task,trained_model)