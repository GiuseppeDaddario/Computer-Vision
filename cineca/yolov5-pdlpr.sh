#!/bin/bash
#SBATCH --job-name=Y-PDLPR
#SBATCH --time=5:00:00                       
#SBATCH --nodes=1                             
#SBATCH --ntasks-per-node=1                   
#SBATCH --cpus-per-task=32                    
#SBATCH --gres=gpu:4                          
#SBATCH --partition=boost_usr_prod           
#SBATCH --qos=normal                        
#SBATCH --output=cineca/logs/Y-PDLPR.out
#SBATCH --error=cineca/logs/Y-PDLPR.err
#SBATCH --account=try25_navigli          


module load cuda/12.1
module load python/3.11

source $SCRATCH/ComputerVision/bin/activate 

cd $SLURM_SUBMIT_DIR 

srun python -m torch.distributed.launch --nproc_per_node=4 cineca/codes/yolov5_pdlpr.py