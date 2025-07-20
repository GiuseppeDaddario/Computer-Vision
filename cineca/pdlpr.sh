#!/bin/bash
#SBATCH --job-name=pdlpr
#SBATCH --time=24:00:00                       
#SBATCH --nodes=1                             
#SBATCH --ntasks-per-node=1                   
#SBATCH --cpus-per-task=32                    
#SBATCH --gres=gpu:4                          
#SBATCH --partition=boost_usr_prod           
#SBATCH --qos=normal                        
#SBATCH --output=cineca/logs/pdlpr.out
#SBATCH --error=cineca/logs/pdlpr.err
#SBATCH --account=try25_navigli          


module load cuda/12.1
module load python/3.11

source $SCRATCH/ComputerVision/bin/activate 

cd $SLURM_SUBMIT_DIR 

srun python -m torch.distributed.launch --nproc_per_node=4 main.py