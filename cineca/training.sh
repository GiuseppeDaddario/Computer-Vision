#!/bin/bash
#SBATCH --job-name=training
#SBATCH --time=04:00:00                       
#SBATCH --nodes=1                             
#SBATCH --ntasks-per-node=1                   
#SBATCH --cpus-per-task=8                     
#SBATCH --gres=gpu:2                          
#SBATCH --partition=boost_usr_prod           
#SBATCH --qos=normal                        
#SBATCH --output=cineca/logs/training.out
#SBATCH --error=cineca/logs/training.err
#SBATCH --account=try25_navigli          


module load cuda/12.1
module load python/3.11

source $SCRATCH/mnlp/bin/activate 

cd $SLURM_SUBMIT_DIR 

python cineca/codes/training.py
