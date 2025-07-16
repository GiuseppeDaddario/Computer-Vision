#!/bin/bash
#SBATCH --job-name=pproc
#SBATCH --time=04:00:00                     
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32     
#SBATCH --partition=boost_usr_prod 
#SBATCH --qos=normal
#SBATCH --output=cineca/logs/pproc.out
#SBATCH --error=cineca/logs/pproc.err
#SBATCH --account=try25_navigli

module load cuda/12.1
module load python/3.11

source ~/mnlp/bin/activate 

cd $SLURM_SUBMIT_DIR 

python cineca/codes/preprocessing.py