#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:36:44 2023

@author: lihao
"""
import numpy as np

#task=np.load('saved_model/task_sali200.npy',allow_pickle=True).item()
task=np.load('/global/homes/h/haoli510/Force_design/saved_model/task_sali200.npy',allow_pickle=True).item()
import os

# Load the molecular positions from the .npz file
#data = np.load('molecular_positions.npz')
positions = task['R_train']
z  = ['C','C','C','C','C','C','C','O','H','O','H','O','H','H','H','H']

# Set up the QChem input file template

template = """
$molecule
 0 1
{positions}
$end

$rem
jobtype                 OPT
GEOM_OPT_COORDS         0
basis                   6-31G
TSVDW                   TRUE
symmetry                false
sym_ignore              true
method                  PBE
aimd_thermostat         nose_hoover
aimd_init_veloc         thermal
aimd_temp               450        !in K - initial conditions AND thermostat
$end
"""

# Loop through the molecules and generate the QChem input files
n_molecule = 2
for i in range(n_molecule):
    # Generate the input file content for this molecule
    
    molecule_positions = positions[i,:,:]
    positions_str = '\n'.join([f'{z[j]} {molecule_positions[j,0]} {molecule_positions[j,1]} {molecule_positions[j,2]}' for j in range(positions.shape[1])])
    molecule_input = template.format(positions=positions_str)

    #molecule_input = template.format(positions=f'{z[j]}\n' + '\n'.join([f'{j+1} {molecule_positions[j][0]} {molecule_positions[j][1]} {molecule_positions[j][2]}' for j in range(positions.shape[1])]))
    
    # Write the input file to disk
    with open(f'molecule_sali_{i+1}.in', 'w') as f:
        f.write(molecule_input)
    
    # Submit the QChem job for optimization
    #os.system(f'qchem molecule_{i+1}.in molecule_{i+1}.out')
