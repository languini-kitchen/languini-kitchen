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
import torch.optim.lr_scheduler as lr_scheduler


def lr_exponential_decay(step, max_steps,
                         max_lr = 0.01,
                         min_lr = 0.001):
  """Exponential decay from max_lr to min_lr over max_steps.
  Continues to decay at the same rate after max_steps.
  Args:
    step: The current training step.
    max_steps: The step value at the end of training.
    max_lr: LR to output at step 0.
    min_lr: LR to output at max_steps.
  Returns:
    The learning rate for the current step.
  """
  assert max_lr > min_lr

  lrate = max_lr * np.power(min_lr / max_lr, step / float(max_steps))
  return lrate


def lr_cosine_decay(step, max_steps,
                    max_lr=0.01,
                    min_lr=0.001,
                    decay_after=True,
                    spike_steps=0,
                    spike_lr=0.0):
  """Cosine decay function. """
  assert max_lr > min_lr
  
  pi = float(torch.pi)
  step_ramp = min(step, max_steps) / max_steps  # ramp: 0 to 1.0.

  lrate = (1 + np.cos(pi * step_ramp)) * 0.5   # ranges from 1 to 0.
  lrate = min_lr + lrate * (max_lr - min_lr)

  if spike_steps > 0 and spike_lr > 0.0:
    assert spike_lr > max_lr
    spike_lrate = spike_lr * ((spike_steps - step) / spike_steps)
    lrate = np.maximum(lrate, spike_lrate)

  if decay_after:
    exp_lrate = lr_exponential_decay(step, max_steps,
                                     max_lr=2*min_lr, min_lr=min_lr)
    lrate = np.where(step < max_steps, lrate, exp_lrate)
  return lrate


class CosineLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_lr, min_lr, max_steps, decay_after, spike_steps=0, spike_lr=0.1):
        self.curr_step = 0
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_steps = max_steps
        self.decay_after = decay_after
        self.spike_steps = spike_steps
        self.spike_lr = spike_lr        
        super(CosineLR, self).__init__(optimizer)

    def get_lr(self):
        self.curr_step += 1

        t = 0
        if self.curr_step > t:
          new_lr = lr_cosine_decay(step=self.curr_step - t, 
                                  max_steps=self.max_steps - t,
                                  max_lr=self.max_lr,
                                  min_lr=self.min_lr,
                                  decay_after=self.decay_after,
                                  spike_steps=self.spike_steps,
                                  spike_lr=self.spike_lr)
        else:
           new_lr = self.max_lr

        warmup_ramp = min(self.curr_step, self.warmup_steps) / self.warmup_steps  # ramp: 0 to 1.0.
        return [warmup_ramp * new_lr for _ in self.base_lrs]

class Warmup(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps=500):
        self.curr_step = 0
        self.warmup_steps = warmup_steps
        super(Warmup, self).__init__(optimizer)

    def get_lr(self):
        self.curr_step += 1
        if self.curr_step <= self.warmup_steps:
          step_ramp = min(self.curr_step, self.warmup_steps) / self.warmup_steps  # ramp: 0 to 1.0.
          return [lr * step_ramp for lr in self.base_lrs]
        else:
           return [pg['lr'] for pg in self.optimizer.param_groups]