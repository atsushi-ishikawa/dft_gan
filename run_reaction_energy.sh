#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -q all.q
#$ -pe openmpi32 32
#$ -l hostname=vodka??

ID=$1

python calc_reaction_energy.py --id $ID --calculator vasp 1> stdout$$.txt 2> stderr$$.txt 
#python calc_reaction_energy.py --id $ID --calculator emt 1> stdout$$.txt 2> stderr$$.txt 
