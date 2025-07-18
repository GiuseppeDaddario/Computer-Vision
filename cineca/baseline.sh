#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --time=24:00:00                       
#SBATCH --nodes=1                             
#SBATCH --ntasks-per-node=1                   
#SBATCH --cpus-per-task=32                    
#SBATCH --gres=gpu:4                          
#SBATCH --partition=boost_usr_prod           
#SBATCH --qos=normal                        
#SBATCH --output=cineca/logs/baseline.out
#SBATCH --error=cineca/logs/baseline.err
#SBATCH --account=try25_navigli          


module load cuda/12.1
module load python/3.11

source $SCRATCH/ComputerVision/bin/activate 

cd $SLURM_SUBMIT_DIR 

python cineca/codes/baseline.py