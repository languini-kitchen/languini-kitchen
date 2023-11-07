# Copyright 2023 The Languini Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch    
import torch.distributed as dist


LOCAL_RANK = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else None
WORLD_SIZE = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else None
WORLD_RANK = int(os.environ['RANK']) if 'RANK' in os.environ else None


def get_device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{LOCAL_RANK}")
    else:
        return torch.device("cpu")


def init_distributed():
    device = get_device()
    if device.type == "cuda":
        backend = 'nccl'
        # this will make all .cuda() calls work properly
        torch.cuda.set_device(device)
    else:
        # no nccl available
        backend = "gloo"

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    print(f"Start running distributed data-parallel experiment on global rank {WORLD_RANK} and local rank {LOCAL_RANK}.")

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()

    return device


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def mprint(txt):
    """Print but only on main device."""
    if is_main_process():
        print(txt)
