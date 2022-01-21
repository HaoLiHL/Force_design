## Work in Progress (WIP) To-do list
* add workflow that conducts simulation after initial geometry prediction.

## WIP log <br>
01/21/2022 <br>
In test_os.py, the simulation I/O is implemented for a given set of predictions.

01/20/2022 <br>
Finish a preliminary simulation input creation and read the energy from simulation.


12/06/2021<br>
update.py contains core python scripts of PPGP in learning structure-force relation and predict next possible geometry for target force.<br>
From the predicted structure, script that converts the structure to simulator and run would be needed.




# Overview
This repository provides workflow to find the specific molecular geometry for target force (e.g., geometry minimization). First, given the name and other settings of the molecule, we want the simulator to generate a dataset such as "H2C0_mu.npz" contains position, force and energy

Second, the update.py will generate a npy file such as "R_design.npy", which contains some molecule's configuration. We need simulator to generate the force and energy for those molecules.


Algorithm:

i) Input: molecule name
ii) Simulator: return the dataset, which serve as input for inverse_force.py

E.g. dataset_uracil.npz = generate_data("uracil", 'other setting")


while achieve the target property:
  iii) train the model based on the current dataset
  
  iv) update.py return a dataset named "R_design.npy" contains the position of molecules.
  
  v) simulator generate an dataset such as "H2C0_mu.npz" contains position, force and energy for dataset from step iv) and update the training dataset

E.g. dataset_uracil_update.npz = generate_data_given_R("R_design.npy")
