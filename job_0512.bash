#!/bin/bash -l
#SBATCH -q regular
#SBATCH -N 8
#SBATCH -C haswell
#SBATCH -t 00:30:00

module load qchem

module load python

python3 example.py > output_job0512_r.txt
