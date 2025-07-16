#!/bin/bash
#SBATCH --job-name=unzipping
#SBATCH --time=04:00:00                     
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8     
#SBATCH --partition=boost_usr_prod 
#SBATCH --qos=normal
#SBATCH --output=cineca/logs/unzipping.out
#SBATCH --error=cineca/logs/unzipping.err
#SBATCH --account=try25_navigli


cd /leonardo/home/userexternal/gdaddari/Computer-Vision/dataset

echo "Inizio decompressione: $(date)"

echo "Uso xz multithread con 8 thread"
xz -T8 -dc CCPD2019.tar.xz | tar -xv

echo "Decompressione completata: $(date)"