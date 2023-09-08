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

import torch
import numpy as np
import pprint as pp
import torch.nn.functional as F

from tqdm import tqdm
from languini.common_lib import common_utils


class MemoryUsage():
    """Context Manager for measuring memory usage."""
    def __init__(self, device):
        self.device = device
         
    def __enter__(self):
        self.begin_mem = torch.cuda.memory_allocated(self.device)
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass
    
    def difference(self):
        curr = torch.cuda.memory_allocated(self.device)
        return curr - self.begin_mem


def throughput_test(config, model, only_forward=False, warmup=5, rounds=15):
    """Loads the model on to config.device and performs measurements such as memory requirements and iterations per second."""
    c = config

    scaler = torch.cuda.amp.GradScaler()
    opt = torch.optim.Adam(model.parameters(), lr=c.max_lr, betas=(0.9, 0.95), eps=1e-09)

    with torch.cuda.amp.autocast():
        # move model to gpu memory and measure usage
        with MemoryUsage(c.device) as memory:
            model = model.to(c.device)
            model_memory_mb = memory.difference() / (1024 ** 2)
        
        def get_batch():
            data = torch.tensor(np.random.choice(a=range(c.vocab_size), size=(c.train_batch_size, c.seq_len + 1), replace=True)).to(c.device)
            batch_x = data[:,:c.seq_len]
            batch_y = data[:,1:]
            return batch_x, batch_y
        
        parameter_count = common_utils.get_total_tensor_size(list(model.parameters()))
        non_embedding = parameter_count - (c.h_dim * c.vocab_size)
        state = model.get_init_state(batch_size=c.train_batch_size, device=c.device)
        
        def step(inputs, targets, state):
            logits, _ = model(inputs, state)
            logits = logits.reshape(c.train_batch_size * c.seq_len, c.vocab_size)
            targets = targets.reshape(c.train_batch_size * c.seq_len)
            avg_loss = F.cross_entropy(input=logits, target=targets).reshape((-1,))
            
            if not only_forward:
                opt.zero_grad(set_to_none=True)
                scaler.scale(avg_loss).backward()
                scaler.step(opt)
                scaler.update()
        
        # measure memory allocation
        with MemoryUsage(c.device) as memory:
            inputs, targets = get_batch()
            step(inputs, targets, state)
            step_memory_alloc_mb = memory.difference() / (1024 ** 2)
        
        # measure time for one step
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        durations = []
        
        for round_i in tqdm(range(rounds + warmup)):
            # load fresh data everytime to prevent caching
            inputs, targets = get_batch()
            # only measure the time per step, excluding copying data over to the accelerator
            start.record()
            step(inputs, targets, state)
            end.record()
            torch.cuda.synchronize()
            if round_i > warmup:
                durations.append(start.elapsed_time(end))
        
        avg_duration_ms = np.mean(durations)
        std_duration_ms = np.std(durations)
        avg_iter_per_second = 1_000 / avg_duration_ms
        std_iter_per_second = np.std(1_000 / np.array(durations))
        device_name = torch.cuda.get_device_name(c.device)
        class_name = model.__class__
        token_count = c.train_batch_size * c.seq_len
        tokens_per_second = round(avg_iter_per_second * token_count)
    
    return {
        "class": class_name,
        "config": pp.pformat(config),
        "performance test": "forward only" if only_forward else "forward and backward",
        "number of parameters": parameter_count,
        "number of non-embedding parameters": non_embedding,
        "device": device_name,
        "model memory usage megabytes": model_memory_mb,
        "step memory usage in megabytes": step_memory_alloc_mb,
        "tokens per batch": token_count,
        "avg step duration in ms": avg_duration_ms,
        "std step duration in ms": std_duration_ms,
        "avg iterations per second": avg_iter_per_second,
        "std iterations per second": std_iter_per_second,
        "tokens per second": tokens_per_second,
    }