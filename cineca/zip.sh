#!/bin/bash
#SBATCH --job-name=zip
#SBATCH --time=04:00:00                     
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8     
#SBATCH --partition=boost_usr_prod 
#SBATCH --qos=normal
#SBATCH --output=cineca/logs/zip.out
#SBATCH --error=cineca/logs/zip.err
#SBATCH --account=try25_navigli


cd /leonardo/home/userexternal/gdaddari/Computer-Vision/dataset/CCPD2019

echo "Inizio creazione zip: $(date)"

# Trova i primi 20000 file e li aggiunge allo zip
find ccpd_base/ -type f -iname "*.jpg" | head -n 20000 > file_list.txt

# Comprime i file trovati nel file "samuele.zip"
zip -@ samuele.zip < file_list.txt

echo "Zip completato: $(date)"