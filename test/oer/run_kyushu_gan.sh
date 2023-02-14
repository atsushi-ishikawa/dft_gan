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

LBL=$$

/home/usr6/m70286a/dft_gan/test/oer/sub2.sh 1> stdout_gan_$LBL.txt 2> stderr_gan_$LBL.txt

