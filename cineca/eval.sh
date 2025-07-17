#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --time=04:00:00                       
#SBATCH --nodes=1                             
#SBATCH --ntasks-per-node=1                   
#SBATCH --cpus-per-task=8                     
#SBATCH --gres=gpu:2                          
#SBATCH --partition=boost_usr_prod           
#SBATCH --qos=normal                        
#SBATCH --output=cineca/logs/eval.out
#SBATCH --error=cineca/logs/eval.err
#SBATCH --account=try25_navigli          


module load cuda/12.1
module load python/3.11

source $SCRATCH/ComputerVision/bin/activate 

cd /leonardo/home/userexternal/gdaddari/Computer-Vision/src/YOLO/yolov5

# Esegui rilevamento
COLUMNS=80 PYTHONWARNINGS="ignore::FutureWarning" python detect.py \
  --weights /leonardo/home/userexternal/gdaddari/Computer-Vision/src/YOLO/runs/train4/weights/best.pt \
  --source /leonardo_scratch/large/userexternal/gdaddari/dataset/CCPD_YOLO/ccpd_challenge/images/test \
  --img 640 \
  --conf 0.25 \
  --iou 0.45 \
  --device 0 \
  --save-txt \
  --save-conf \
  --project /leonardo/home/userexternal/gdaddari/Computer-Vision/src/YOLO/runs \
  --name detect \
  --exist-ok
