import numpy as np
from utils import AFF


# load the dataset contains the geometry information and the force-fields

#dataset=np.load('./dataset/aspirin_ccsd-train.npz')
#dataset=np.load('uracil_dft_mu.npz')
dataset=np.load('H2CO_mu.npz')
AFF_train=AFF.AFFTrain()

n_train=100

#create the task file contains the training, validation and testing dataset 
task=AFF_train.create_task(train_dataset=dataset, 
                            n_train = n_train ,
                            valid_dataset=dataset,
                            n_valid=50,
                            n_test=50,
                            lam = 1e-15)
#np.save('uracil_task',task)
#task = np.load('uracil_task.npy',allow_pickle=True).item()
#trained_model = np.load('uracil_trained_model.npy',allow_pickle = True).item()
# start training the model based on the training dataset
trained_model = AFF_train.train(task,sig_candid_F = np.arange(10,100,30))
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

initial=3
#n_sam=11
#R_target1=np.empty((n_sam,12,3))
#F_predict=np.empty((n_sam,12,3))
#cost=np.empty((n_sam,1))

#np.array(R_design),R_val_atom_last,F_hat,record,cost_SAE
R_design,R_val_atom_last,F_hat,record,cost_SAE = AFF_train.inverseF(task,
                                                                    trained_model,                                                                
                                                                    initial,
                                                                    F_target,
                                                                    lr=1e-15)

AFF.compile_scirpts_for_physics_based_calculation_IO(R_design)
# 
# atomic number shall be defined in the beginning of the program once as well
atomic_number = dataset['z']
# 
# for-loop
new_E, new_F = AFF.run_physics_baed_calculation(R_design, atomic_number)
print('new_E,new_F ',new_F)

