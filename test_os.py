# this script demonstates how to carry the calculation for 
# a set of saved database like R_design.npy


import numpy as np
import subprocess
import os
import math
import pickle

# # by running this command, you don't suspend the python script from running
# test = subprocess.Popen(["echo", "test"])
# # this script would have the python script wait till the os interaction finishes
# test.wait()

# read element table list
with open('./scripts/element_table', 'rb') as fp:
	element_table = pickle.load(fp)


# this command can be removed once merge into the main program
R_design = np.load('R_design.npy')

atomic_number = np.load('atomic_number.npy')



# it seems the number of predicted molecules is not directly readable in the update.py
# it seems there is no variables to record how many atoms in one molecules
[num_molecules, num_atoms,*_] = R_design.shape

simulator_input_filename = "simulator_input.dat"
simulator_output_filename = "simulator_output.dat"
num_parallel_core = 32

simulation_failed_string = 'SCF failed to converge'
temp_energy_filename = "temp_energy.dat"
energy_filename = "energy.dat"
temp_force_filename = "temp_force.dat"
force_filename = "force.dat"


energy_converter_script = './scripts/script_E.c'
energy_converter_script_path = './scripts/script_E'
force_converter_script = './scripts/script_F.c'
cp_force_converter_script = './scripts/cp_script_F.c'
force_converter_script_path = './scripts/script_F'

command_compile_E_convertor = ["gcc", energy_converter_script, "-o", energy_converter_script_path]
compile_E = subprocess.Popen(' '.join(command_compile_E_convertor), shell=True)
compile_E.wait()


command_cp_F_convertor = ["cp", force_converter_script, cp_force_converter_script]
cp_F_convertor = subprocess.Popen(' '.join(command_cp_F_convertor), shell=True)
cp_F_convertor.wait()
replace_content = ["\"s/NUMATOM/", str(num_atoms), "/g\""]
command_replace_num_atom_F_convertor = ["sed", "-i", ''.join(replace_content), cp_force_converter_script]
replace_F = subprocess.Popen(' '.join(command_replace_num_atom_F_convertor), shell=True)
replace_F.wait()
command_compile_F_convertor = ["gcc", cp_force_converter_script, "-o", force_converter_script_path, "-lm"]
compile_F = subprocess.Popen(' '.join(command_compile_F_convertor), shell=True)
compile_F.wait()




E_new_simulation = np.zeros((num_molecules,))
F_new_simulation = np.zeros((num_molecules,num_atoms,3))







for index_run in range(num_molecules):
# for index_run in range(2):

	# in principle when writing the calculation file the loop should not assume
	# the element in the molecule, however not sure where the atomic number for
	# each element in the molecule is recorded, the following session would only
	# work for specific situation in H2CO

	# maybe we need a switch to choose which type of simulator input needed for 
	# different purpose
	# in this section, we only have qchem input implemented
	with open(simulator_input_filename, 'wt') as input_write:
		input_write.write("$molecule\n")
		input_write.write("0 1\n")

		for index_atom in range(len(atomic_number)):
			input_write.write("%s\t" % element_table[atomic_number[index_atom]-1])

			for i_dimension in range(3):
				# print(R_design[index_run, 0, i_dimension])
				input_write.write("%.8lf\t" % R_design[index_run, index_atom, i_dimension])

			input_write.write("\n")




			
		input_write.write("$end\n")
		input_write.write("\n")
		input_write.write("$rem\n")
		input_write.write("jobtype                force\n")
		input_write.write("exchange               HF\n")
		input_write.write("correlation            mp2\n")
		input_write.write("basis                  aug-cc-pVTZ\n")
		input_write.write("SCF_CONVERGENCE 11\n")
		input_write.write("symmetry false\n")
		input_write.write("sym_ignore true\n")
		input_write.write("$end\n")




	# submit jobs to calculation
	command_run_simulation = ["qchem", "-slurm", "-nt", str(num_parallel_core), simulator_input_filename, ">", simulator_output_filename]
	run_simulator = subprocess.Popen(' '.join(command_run_simulation), shell=True)
	run_simulator.wait()

	signal_simulation_success = 1
	with open(simulator_output_filename, "rt") as read_ouput:
	# with open('failed.out', 'rt') as read_ouput:
		for line in read_ouput:
			# print(line)
			if simulation_failed_string in line:
				signal_simulation_success = 0
				print('simulation failed')
				break

	if signal_simulation_success == 1:
		print('simulation succeed')

		# read energy into the matrix
		command_grep_energy = ["grep", "\"Convergence criterion met\"", simulator_output_filename, ">", temp_energy_filename]
		grep_energy = subprocess.Popen(' '.join(command_grep_energy), shell=True)
		grep_energy.wait()
		command_conv_energy = [energy_converter_script_path, temp_energy_filename, energy_filename]
		conv_energy = subprocess.Popen(' '.join(command_conv_energy), shell=True)
		conv_energy.wait()
		single_energy = np.loadtxt(energy_filename)
		os.remove(temp_energy_filename)
		os.remove(energy_filename)
		if np.any(single_energy) is False:
			print('!!!!!!!! no energy read')
		else:
			E_new_simulation[index_run] = single_energy


		command_grep_force = ["grep", "-A",  str(math.ceil(num_atoms/6)*4), "\"Full Analytical Gradient of MP2 Energy\"", simulator_output_filename, ">", temp_force_filename]
		grep_force = subprocess.Popen(' '.join(command_grep_force), shell=True)
		grep_force.wait()
		command_force_energy = [force_converter_script_path, temp_force_filename, force_filename]
		conv_force = subprocess.Popen(' '.join(command_force_energy), shell=True)
		conv_force.wait()
		single_force = np.loadtxt(force_filename)
		os.remove(temp_force_filename)
		os.remove(force_filename)
		if np.any(single_force) is False:
			print('!!!!!!!! no force read')
		else:
			if single_force.shape is not F_new_simulation[0,:,:]:
				F_new_simulation[index_run,:,:] = single_force
			else:
				print("!!!!!!! wrong dimension in force extraction")
				print("!!!!!!! please check")


		
		# print(single_energy)
		# print(single_energy.shape)



		# grep_energy = subprocess.Popen(["cat", temp_energy_filename, "|", "grep", "\"Convergence criterion met\"", simulator_output_filename])


	# else:
		# what if simulation failed?
		# what if simulation failed?
		# what if simulation failed?
		# what data shall we return to force then?
		# what data shall we return to force then?
		# what data shall we return to force then?
		# what data shall we return to force then?
		# shall we abandon both structure and force in the next training phase
		# shall we abandon both structure and force in the next training phase
		# shall we abandon both structure and force in the next training phase
		# shall we abandon both structure and force in the next training phase
		# shall we abandon both structure and force in the next training phase



print(E_new_simulation)
print(F_new_simulation)

with open('E_feedback.npy', 'wb') as f:
    np.save(f, E_new_simulation)

with open('F_feedback.npy', 'wb') as f:
    np.save(f, F_new_simulation)


