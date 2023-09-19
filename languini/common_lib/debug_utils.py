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


def check(tensor, shape):
    """ Checks the shape of the tensor for better code redability and bug prevention. """
    if tensor is None:
        return
    
    assert isinstance(tensor, torch.Tensor), "SHAPE GUARD: tensor is not torch.Tensor!"
    tensor = tensor.detach()  # necessary for torch.compile
    tensor_shape = list(tensor.shape)

    assert isinstance(shape, list) or isinstance(shape, tuple), "shape arg has to be a tuple/list!"
    assert len(shape) == len(tensor_shape), f"SHAPE GUARD: tensor shape {tensor_shape} not the same length as {shape}"
    
    for idx, (a, b) in enumerate(zip(tensor_shape, shape)):
        if b <= 0:
            continue  # ignore -1 sizes
        else:
            assert a == b, f"SHAPE GUARD: at pos {str(idx)}, tensor shape {tensor_shape} does not match {shape}"


def log_stats_and_dist(tensor, prefix, log):
    """Logs distribution statistics of a tensor."""
    if log is None:
        return

    logger, step = log

    # if this is not the master processor we do have a log argument but no logger
    if logger is None:
        return

    # compute stats on gpu
    #tensor = tensor.detach().cpu().float()
    tensor_mean = tensor.mean().float()
    tensor_max = tensor.max().float()
    tensor_min = tensor.min().float()
    tensor_l2 = torch.norm(tensor).float()
    
    # write as scalar
    logger.log(
        {
            f"{prefix}/mean": tensor_mean.cpu(),
            f"{prefix}/max": tensor_max.cpu(),
            f"{prefix}/min": tensor_min.cpu(),
            f"{prefix}/l2": tensor_l2.cpu(),
        },
        step=step,
    )


def log_global_stats_and_dist(tensors, prefix, log):
    """Logs distribution statistics of a list of tensors such as all model parameters."""
    if log is None:
        return

    logger, step = log

    # if this is not the master processor we do have a log argument but no logger
    if logger is None:
        return
    
    # compute stats on device
    global_numel = 0
    global_sum = 0
    global_pow2sum = 0
    global_max = -torch.inf
    global_min = torch.inf
    for tensor in tensors:
        # compute stats on gpu
        tensor = tensor.detach().float()
        global_numel += tensor.numel()  # already int not tensor
        global_sum += tensor.sum().cpu()
        global_pow2sum +=  tensor.pow(2).sum().cpu()
        global_max = max(global_max, tensor.max().cpu())
        global_min = min(global_min, tensor.min().cpu())
    
    global_mean = global_sum / global_numel
    global_l2 = torch.sqrt(global_pow2sum)
    
    # write as scalar
    logger.log(
        {
            f"{prefix}/mean": global_mean,
            f"{prefix}/max": global_max,
            f"{prefix}/min": global_min,
            f"{prefix}/l2": global_l2,
        },
        step=step,
    )


def log_gradient_stats(model, prefix, log):
    """Logs gradient stats per weight tensor and globally. """
    if log is None:
        return

    with torch.no_grad():
        global_grads = []
        for name, p in model.named_parameters():
            if not hasattr(p.grad, 'data'):
                print(f"WARNING: no gradient for parameter {name}")
                continue
            grad = p.grad.data.detach()
            global_grads.append(grad)
            shape = "x".join(f"{d}" for d in grad.shape)
            name = f"{name} {shape}"
            log_stats_and_dist(grad, f"{prefix}/{name}", log)
        
        log_global_stats_and_dist(global_grads, f"{prefix}/global", log)


def log_weight_stats(model, prefix, log):
    """Logs weight stats per weight tensor and globally. """
    if log is None:
        return
    
    with torch.no_grad():
        weights = []
        for name, p in model.named_parameters():
            weight_tensor = p.data.detach()
            weights.append(weight_tensor)
            shape = "x".join(f"{d}" for d in weight_tensor.shape)
            name = f"{name} {shape}"
            log_stats_and_dist(weight_tensor, f"{prefix}/{name}", log)
        
        log_global_stats_and_dist(weights, f"{prefix}/global", log)


def step_and_log_diff(scaler, opt, model, prefix, log):
    with torch.no_grad():
        pre_p = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        diffs = {name: p.detach().clone() - pre_p[name] for name, p in model.named_parameters() if p.requires_grad}
        
        for name, diff in diffs.items():
            shape = "x".join(f"{d}" for d in diff.shape)
            name = f"{name} {shape}"
            log_stats_and_dist(diff, f"{prefix}/{name}", log)
        
        log_global_stats_and_dist(diffs.values(), f"{prefix}/global", log)
