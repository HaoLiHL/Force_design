#!/bin/bash -l
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 00:20:00

module load qchem 

module load python

python3 example.py > output_job0512.txt


