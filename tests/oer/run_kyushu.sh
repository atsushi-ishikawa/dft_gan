#!/bin/bash
#PJM -L "rscunit=ito-a"
#PJM -L "rscgrp=ito-a-oc170117"
#PJM -L "vnode=1"
#PJM -L "vnode-core=36"
#PJM -L "elapse=10:00:00"
#PJM -j
#PJM -X

NUM_NODES=${PJM_VNODES}
NUM_CORES=36
NUM_PROCS=`expr $NUM_CORES "*" $NUM_NODES`

module load intel/2017

export I_MPI_PERHOST=$NUM_CORES
export I_MPI_FABRICS=shm:ofa

export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=/bin/pjrsh
export I_MPI_HYDRA_HOST_FILE=${PJM_O_NODEINF}

vasp_script="/home/usr6/m70286a/ase/run_vasp.py"
PRG=${HOME}/vasp/vasp.5.4.4/bin/vasp_std
echo "import os" > $vasp_script
echo "exitcode = os.system(\"mpirun -np ${NUM_PROCS} ${PRG}\")" >> $vasp_script

LBL=$$
unique_id=$unique_id
reactionfile="oer.txt"

echo "unique_id=$unique_id"

python ../../calc_reaction_energy.py --reactionfile=$reactionfile --unique_id=$unique_id --calculator=vasp --npar=6 1> stdout_$LBL.txt 2> stderr_$LBL.txt 

