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
import time
import torch
import psutil
import numpy as np
import sentencepiece as spm

from languini.common_lib.debug_utils import check
from languini.common_lib.parallel_utils import mprint


def assert_entries_exist(map, keys):
  """Raises an attribute error if any on the keys does not exist. """
  for k in keys:
    if k not in map.__dict__.keys():
      raise AttributeError("Necessary parameter {} is missing!".format(k))


def check_config(config, defaults):
   """Checks if the config contains the default values. Sets them if the default is not None. """
   for key in defaults:
      if key not in config.__dict__.keys():
         if defaults[key]:
            config[key] = defaults[key]
         else:
            raise TypeError(f"Missing one required config entry: {key}")


def get_param_count(model):
  """Returns the total number of trainable parameters. """
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_tokeniser(config):
  vocab_path = os.path.join(os.getcwd(), 'languini/vocabs/spm_models', f"{config.dataset}.model")
  mprint(f"load vocab from: {vocab_path}")
  sp = spm.SentencePieceProcessor()
  if not sp.Load(vocab_path):
      raise Exception("Couldn't load tokeniser.")
  return sp


def load_checkpoint(model, path):
    """Loads a model from a checkpoint on disk. """
    new_state_dict = {}
    with open(path, 'rb') as f:
        checkpoint = torch.load(f)
        model_state_dict = checkpoint["model_state_dict"]
        for key, value in model_state_dict.items():
            new_state_dict[key] = value
        model.load_state_dict(new_state_dict)

        curr_state = checkpoint["curr_state"] if 'curr_state' in checkpoint else None
    return model, curr_state


def print_model_size(model, vocab_size, h_dim, logger):
    line_len = 89
    line_len2 = 25
    mprint('-' * line_len)
    # Native pytorch
    try:
        mprint(model)
    except:
        mprint('Warning: could not print the Native PyTorch model info - probably some module is `None`.')

    # One-by-one layer
    mprint('-' * line_len)
    mprint("Model params:")
    total_params = 0
    module_name = ""
    module_n_params = 0
    for name, param in model.named_parameters():
        if name.find('.') != -1:
            if module_name == "":
                module_name = name[:name.index('.')]
            if module_name != name[:name.index('.')]:
                mprint('=' * line_len2 + f" {module_name} {module_n_params:,} " + '=' * line_len2 + '\n')
                module_name = name[:name.index('.')]
                module_n_params = 0
        else:
            if module_name == "":
                module_name = name
            if module_name != name:
                mprint('=' * line_len2 + f" {module_name} {module_n_params:,} " + '=' * line_len2 + '\n')
                module_name = name
                module_n_params = 0
        n_params = np.prod(param.size())
        module_n_params += n_params
        mprint(f"\t {name} {n_params:,}")
        total_params += n_params
    mprint('=' * line_len2 + f" {module_name} {module_n_params:,}" + '=' * line_len2 + '\n')

    # Total number of trainable params w/ and w/o embedding params
    mprint('-' * line_len)
    params = get_param_count(model)
    params_without_emb = params - 2*(vocab_size * h_dim)
    logger.log(
        {
            "parameter count": params,
            "parameter count without embedding": params_without_emb,
        },
        step=None,
    )
    mprint(f"Total trainable parameters: {params:,}")
    mprint(f"Total trainable parameters without input and output embedding: {params_without_emb:,}")
    mprint('-' * line_len)


def total_correct(logits, target, is_padding, topk=(1,)):
    """
    Computes the accuracy over the k top predictions.

    Args:
      logits: logits (batch_size, vocab_size)
      target: target labels (batch_size)
      is_padding: True where data is padding and should be ignored (batch_size)
      topk: list of integers specifying which top-k to compute
    
    Returns:
      topk_counts: dict of k -> number of correct predictions for that topk (ignoring padding)
    """
    assert logits.shape[0] == target.shape[0], "shape mismatch in dim=0 for logits and targets"
    assert target.shape == is_padding.shape and target.ndim == 1

    topk_counts = {}

    with torch.no_grad():
        maxk = max(topk)
        bsz = logits.shape[0]

        _, preds = logits.topk(k=maxk, dim=1)
        preds = preds.t()
        check(preds, (maxk, bsz))

        # compare
        correct = (preds == target.unsqueeze(0))

        # ignore padding
        correct &= ~is_padding.unsqueeze(0)

        for k in topk:          
          # get top k predictions
          topk_pred = correct[:k]
          topk_pred = topk_pred.reshape(-1).float()
          check(topk_pred, (k * bsz,))

          topk_count = topk_pred.sum(dim=0, keepdim=True)
          check(topk_count, (1,))

          topk_counts[k] = topk_count

        return topk_counts
    

def log_hyperparams(config, logger):
  if logger is not None:
    config_txt = "Experiment Config  \n\n"
    for key, value in config.toDict().items():
      if type(value) in [int, float, bool, str, torch.Tensor]:
        config_txt += f"{key} = {value}  \n"
      else:
        try:
          config_txt += f"{key} = {str(value)}  \n"
        except:
          pass
    logger.log_text("config", config_txt)


class StopWatch:
  """Keeps track of time through pauses. """

  def __init__(self):
    self.total = 0
    self.counter = 0
    self.start_time = 0

  def start(self):
    """Starts the stopwatch or continues where it left of. """
    self.start_time = time.time()
    return self

  def pause(self):
    """Pauses the stopwatch and adds the delta to the total time. """
    if self.start_time != 0:
        self.total += time.time() - self.start_time
        self.start_time = 0
    return self

  def read(self):
    """Outputs the current total time divided by counts and resets the stopwatch. """
    time_passed = 0 if self.start_time == 0 else time.time() - self.start_time
    if self.total + time_passed == 0:
       value = 0
    else:
        value = (self.total + time_passed) / max(self.counter,1)
    self.reset()
    return value
  
  def count(self):
    """Increases counter responsible for averaging. """
    self.counter += 1
    return self

  def reset(self):
    """Resets the total time. Restarts the stopwatch if it was running. """
    if self.start_time != 0:
       self.start()
    self.total = 0
    self.counter = 0
    return self