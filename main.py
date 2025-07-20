import os
import torch
import torch.distributed as dist

## --- PDLPR --- ##
from src.PDLPR.training import PDLPR_training as PDLPR_training_augmentation
from src.PDLPR.inference import PDLPR_inference, all_inference

def setup_ddp():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world_size

def cleanup_ddp():
    dist.destroy_process_group()

def main():
    rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    if rank == 0:
        print("PDLPR Training with DDP ...")

    TRAINING_PATH = os.path.join(os.environ["SCRATCH"], "dataset/CCPD2019/ccpd_base")
    
    PDLPR_training_augmentation(
        TRAINING_PATH, 
        num_epochs=3, 
        batch_size=512, 
        rank=rank, 
        world_size=world_size, 
        device=device
    )

    cleanup_ddp()

if __name__ == "__main__":
    main()