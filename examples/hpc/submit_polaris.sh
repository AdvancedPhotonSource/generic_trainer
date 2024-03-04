#!/bin/sh
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:10:00
#PBS -q debug
#PBS -A APSDataAnalysis
#PBS -l filesystems=home:eagle:grand
#PBS -e /eagle/APSDataAnalysis/alphadiffract/github/AlphaDiffract/code/demos/train_hpc/outputs/
#PBS -o /eagle/APSDataAnalysis/alphadiffract/github/AlphaDiffract/code/demos/train_hpc/outputs/

# Set up software deps:
VENV_DIR="/eagle/projects/APSDataAnalysis/alphadiffract/venvs/alphadiffract"
source "${VENV_DIR}/bin/activate"
export PYTHONPATH="/eagle/projects/APSDataAnalysis/alphadiffract/github/AlphaDiffract/code:$PYTHONPATH"

# Parallelization variables
N_NODES=`wc -l < $PBS_NODEFILE`
N_PROCS_PER_NODE=2
N_TOTAL_PROCS=$(( $N_NODES * $N_PROCS_PER_NODE ))
echo "N_NODES=$N_NODES N_TOTAL_PROCS=$N_TOTAL_PROCS"

cd $PBS_O_WORKDIR

aprun -n $N_TOTAL_PROCS -N $N_PROCS_PER_NODE python ddp_training_with_dummy_data.py 
