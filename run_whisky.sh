#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -q all.q
#$ -pe openmpi24 24
#$ -l hostname=whisky1?

# clean
#rm *.{e,o}[0-9]*
#rm stdout.txt stderr.txt
#rm -rf N2* Pd*

python calc_reaction_energy.py --calculator vasp 1> stdout$$.txt 2> stderr$$.txt 
