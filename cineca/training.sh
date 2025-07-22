#!/bin/bash
#SBATCH --job-name=training
#SBATCH --time=4:00:00                       
#SBATCH --nodes=1                             
#SBATCH --ntasks-per-node=1                   
#SBATCH --cpus-per-task=32                    
#SBATCH --gres=gpu:4                          
#SBATCH --partition=boost_usr_prod           
#SBATCH --qos=normal                        
#SBATCH --output=cineca/logs/training.out
#SBATCH --error=cineca/logs/training.err
#SBATCH --account=try25_navigli          


module load cuda/12.1
module load python/3.11

source $SCRATCH/ComputerVision/bin/activate 

cd /leonardo/home/userexternal/gdaddari/Computer-Vision/src/YOLO/yolov5

COLUMNS=80 PYTHONWARNINGS="ignore::FutureWarning" \
python -m torch.distributed.run --nproc_per_node=4 train.py \
  --data /leonardo/home/userexternal/gdaddari/Computer-Vision/dataset/ccpd_2019.yaml \
  --weights /leonardo/home/userexternal/gdaddari/Computer-Vision/src/YOLO/runs/train/weights/best.pt \
  --batch-size 256 \
  --img 640 \
  --epochs 20 \
  --optimizer Adam \
  --hyp /leonardo/home/userexternal/gdaddari/Computer-Vision/src/YOLO/hyp.yaml \
  --cos-lr \
  --project /leonardo/home/userexternal/gdaddari/Computer-Vision/src/YOLO/runs \
  --name train \
  --cache ram \
  --device 0,1,2,3 \
  --workers 8
