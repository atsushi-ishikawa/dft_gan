#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -q all.q
#$ -pe openmpi24 24
#$ -l hostname=whisky??&!whisky08&!whisky02&!whisky12

ID=$1

python calc_reaction_energy.py --id $ID --calculator vasp 1> stdout$$.txt 2> stderr$$.txt 
#python calc_reaction_energy.py --id $ID --calculator emt 1> stdout$$.txt 2> stderr$$.txt 
