#!/bin/sh 
#------ pjsub option --------#
#PJM -L rscgrp=n22240a
#PJM -L node=1
#PJM -L elapse=20:00:00 
#PJM -g n22240
#PJM -j
#------- Program execution -------#
NUM_NODES=${PJM_VNODES}
NUM_CORES=40
NUM_PROCS=`expr $NUM_NODES "*" $NUM_CORES`

module load intel

vasp_script=${HOME}/ase/run_vasp.py
PRG=${HOME}/vasp/vasp.5.4.4/bin/vasp_std
echo "import os" > $vasp_script
echo "exitcode = os.system(\"mpirun -np ${NUM_PROCS} ${PRG}\")" >> $vasp_script

LBL=$$
unique_id=$unique_id
reactionfile="oer.txt"
encut=400
steps=200
npar=10

echo "unique_id=$unique_id"

python ../../calc_reaction_energy.py --reactionfile=$reactionfile --unique_id=$unique_id --calculator=vasp --npar=$npar --steps=$steps --encut=$encut 1> stdout_$LBL.txt 2> stderr_$LBL.txt 

