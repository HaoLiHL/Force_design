First, given the name and other settings of the molecule, we want the simulator to generate a dataset such as "H2C0_mu.npz" contains position, force and energy

Second, the update.py will generate a npz file such as "R_design.npy", which contains some molecule's configuration. We need simulator to generate the force and energy for those molecules.


Algorithm:

i) Input: molecule name
ii) Simulator: return the dataset, which serve as input for inverse_force.py

while achieve the target property:
  iii) train the model based on the current dataset
  iv) update.py return a dataset named "R_design.npy" contains the position of molecules.
  v) simulator generate an dataset such as "H2C0_mu.npz" contains position, force and energy for dataset from step iv) and update the training dataset

