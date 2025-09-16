#!/bin/bash
#SBATCH --job-name=bash                 # avoid lightning auto-debug configuration
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1             # shall >#GPU to avoid overtime thread distribution
#SBATCH --cpus-per-task=8               # number of OpenMP threads per MPI process
#SBATCH --mem=64G
#SBATCH --time 6-23:59:59               # time limit (D-HH:MM:ss)
#SBATCH --output=logs/runs/%x-%j.out    # output file
#SBATCH --gres=gpu:1

#SBATCH --partition=yss
#SBATCH --nodelist=beren,luthien

##SBATCH --partition=jsteinhardt
##SBATCH --nodelist=saruman

#########################
####### Metadata ########
#########################
dt=$(date '+%d/%m/%Y-%H:%M:%S')
echo "Job started on $(hostname) at ${dt}"
echo "GPU info:"
nvidia-smi

#########################
####### Routine #########
#########################
module load cuda/12.8
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NCCL_DEBUG=INFO

uv run python $@

echo "Job completed at $(date)"
