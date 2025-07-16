#!/bin/bash
#SBATCH --job-name=preprocessing
#SBATCH --time=04:00:00                     
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8     
#SBATCH --partition=boost_usr_prod 
#SBATCH --qos=normal
#SBATCH --output=cineca/logs/preprocessing.out
#SBATCH --error=cineca/logs/preprocessing.err
#SBATCH --account=try25_navigli

module load cuda/12.1
module load python/3.10

source ~/mnlp/bin/activate 

cd cineca/codes

python preprocessing.py