#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -q all.q
#$ -pe openmpi24 24
#$ -l hostname=whisky[0-2]?

ID=$1

python calc_reaction_energy.py --id $ID --calculator vasp 1> stdout$$.txt 2> stderr$$.txt 
