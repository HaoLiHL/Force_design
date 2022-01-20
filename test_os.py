# this script demonstates how to carry the calculation for 
# a set of saved database like R_design.npy


import numpy as np
import subprocess
import os

# # by running this command, you don't suspend the python script from running
# test = subprocess.Popen(["echo", "test"])
# # this script would have the python script wait till the os interaction finishes
# test.wait()


# this command can be removed once merge into the main program
R_design = np.load('R_design.npy')



# it seems the number of predicted molecules is not directly readable in the update.py
# it seems there is no variables to record how many atoms in one molecules
[num_molecules, num_atoms,*_] = R_design.shape

simulator_input_filename = "simulator_input.dat"
simulator_output_filename = "simulator_output.dat"
num_parallel_core = 32

simulation_failed_string = 'SCF failed to converge'
temp_energy_filename = "temp_energy.dat"
energy_filename = "energy.dat"


energy_converter_script = './scripts/script_E.c'
energy_converter_script_path = './scripts/script_E'
command_compile_E_convertor = ["gcc", energy_converter_script, "-o", energy_converter_script_path]
compile_E = subprocess.Popen(' '.join(command_compile_E_convertor), shell=True)
compile_E.wait()




E_new_simulation = np.zeros((num_molecules,))
F_new_simulation = np.zeros((num_molecules,num_atoms,3))







# for index_run in range(num_molecules):
for index_run in range(1):

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
		# this section could be simplified
		# this section could be simplified
		# this section could be simplified
		# this section could be simplified
		# this section could be simplified
		input_write.write("C\t")
		for i_dimension in range(3):
			# print(R_design[index_run, 0, i_dimension])
			input_write.write("%.8lf\t" % R_design[index_run, 0, i_dimension])
		input_write.write("\n")
		input_write.write("O\t")
		for i_dimension in range(3):
			input_write.write("%.8lf\t" % R_design[index_run, 1, i_dimension])
		input_write.write("\n")
		input_write.write("H\t")
		for i_dimension in range(3):
			input_write.write("%.8lf\t" % R_design[index_run, 2, i_dimension])
		input_write.write("\n")
		input_write.write("H\t")
		for i_dimension in range(3):
			input_write.write("%.8lf\t" % R_design[index_run, 3, i_dimension])
		input_write.write("\n")
		# this section could be simplified
		# this section could be simplified
		# this section could be simplified
		# this section could be simplified
		# this section could be simplified
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
	# command_run_simulation = ["qchem", "-slurm", "-nt", num_parallel_core, simulator_input_filename, ">", simulator_output_filename]
	# run_simulator = subprocess.Popen(' '.join(command_run_simulation), shell=True)
	# run_simulator.wait()

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

