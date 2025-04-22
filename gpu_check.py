import multiprocessing as mp
mp.set_start_method("spawn", force=True)

from accelerate import Accelerator
import torch.distributed as dist

acc = Accelerator()
acc.print(f"Rank: {acc.process_index} | Local Rank: {acc.local_process_index} | Device: {acc.device}")

if dist.is_initialized():
    acc.print(f"World size: {dist.get_world_size()} | My rank: {dist.get_rank()}")
else:
    acc.print("torch.distributed not initialized")
