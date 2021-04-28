#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -q all.q
#$ -pe openmpi12 12
#$ -l hostname=whisky1?

python calc_reaction_energy.py --calculator vasp 1> stdout$$.txt 2> stderr$$.txt 
