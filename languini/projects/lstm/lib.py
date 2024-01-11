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

import math
import torch
import torch.nn.functional as F

from torch import nn
from languini.common_lib.debug_utils import check
from languini.common_lib.debug_utils import log_stats_and_dist


def gelu(x):
    """
    Gaussian Error Linear Unit (GELU) 
    https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
  def __init__(self, h_dim, name):
    super().__init__()
    self.name = name
    self.h_dim = h_dim
    self.weight = nn.Parameter(torch.ones(h_dim))
    self.bias = nn.Parameter(torch.zeros(h_dim))

  def forward(self, x, log=None):
    bsz, seqlen, _ = x.shape
    check(x, (bsz, seqlen, self.h_dim))
    
    log_stats_and_dist(x, f"{self.name}.preLN", log)
    y = F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
    log_stats_and_dist(x, f"{self.name}.postLN", log)
    return y


class MultiHeadLSTMCell(nn.Module):
  """Implements an LSTM cell with multiple heads."""
  def __init__(self, seq_len, h_dim, head_dim, n_heads, name):
    super().__init__()
    self.name = name
    self.h_dim = h_dim
    self.head_dim = head_dim
    self.n_heads = n_heads
    self.seq_len = seq_len

    # linear projections for x (done in parallel)
    self.linear_Fx = nn.Linear(h_dim, head_dim * n_heads, bias=True)
    torch.nn.init.normal_(self.linear_Fx.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(self.linear_Fx.bias)

    self.linear_Ix = nn.Linear(h_dim, head_dim * n_heads, bias=True)
    torch.nn.init.normal_(self.linear_Ix.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(self.linear_Ix.bias)

    self.linear_Zx = nn.Linear(h_dim, head_dim * n_heads, bias=True)
    torch.nn.init.normal_(self.linear_Zx.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(self.linear_Zx.bias)

    self.linear_Ox = nn.Linear(h_dim, head_dim * n_heads, bias=True)
    torch.nn.init.normal_(self.linear_Ox.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(self.linear_Ox.bias)

    # linear projections of the state (done in sequence)
    self.linear_Fh = nn.Linear(n_heads * head_dim, head_dim, bias=True)
    torch.nn.init.normal_(self.linear_Fh.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(self.linear_Fh.bias)

    self.linear_Ih = nn.Linear(n_heads * head_dim, head_dim, bias=True)
    torch.nn.init.normal_(self.linear_Ih.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(self.linear_Ih.bias)

    self.linear_Zh = nn.Linear(n_heads * head_dim, head_dim, bias=True)
    torch.nn.init.normal_(self.linear_Zh.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(self.linear_Zh.bias)

    self.linear_Oh = nn.Linear(n_heads * head_dim, head_dim, bias=True)
    torch.nn.init.normal_(self.linear_Oh.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(self.linear_Oh.bias)

    # additional linear layer to project all heads back to h_dim in case of multiple heads (as in attention)
    if self.n_heads > 1:
      self.linear_post_head = nn.Linear(head_dim * n_heads, h_dim, bias=True)
      torch.nn.init.normal_(self.linear_post_head.weight, mean=0.0, std=0.02)
      torch.nn.init.zeros_(self.linear_post_head.bias)
    
  def forward(self, f_in, i_in, z_in, o_in, state, log=None):
    bsz, _, _ = f_in.shape
    c, h = state
    check(c, (bsz, self.n_heads, self.head_dim))
    check(h, (bsz, self.n_heads, self.head_dim))

    check(f_in, (bsz, self.seq_len, self.h_dim))
    check(i_in, (bsz, self.seq_len, self.h_dim))
    check(z_in, (bsz, self.seq_len, self.h_dim))
    check(o_in, (bsz, self.seq_len, self.h_dim))

    # compute the gates given x in parallel
    fx = self.linear_Fx(f_in)  # forget gate
    ix = self.linear_Ix(i_in)  # input gate
    zx = self.linear_Zx(z_in)  # cell update
    ox = self.linear_Ox(o_in)  # output gate

    # log activations
    log_stats_and_dist(fx, f"{self.name}.fx", log)
    log_stats_and_dist(ix, f"{self.name}.ix", log)
    log_stats_and_dist(zx, f"{self.name}.zx", log)
    log_stats_and_dist(ox, f"{self.name}.ox", log)

    fx = fx.view(bsz, self.seq_len, self.n_heads, self.head_dim).transpose(1,2)
    ix = ix.view(bsz, self.seq_len, self.n_heads, self.head_dim).transpose(1,2)
    zx = zx.view(bsz, self.seq_len, self.n_heads, self.head_dim).transpose(1,2)
    ox = ox.view(bsz, self.seq_len, self.n_heads, self.head_dim).transpose(1,2)
    check(fx, (bsz, self.n_heads, self.seq_len, self.head_dim))

    # iterate over the sequence
    outputs = []
    for idx in range(self.seq_len):
      check(h, (bsz, self.n_heads, self.head_dim))
      h = h.view(bsz, self.n_heads * self.head_dim * 2)
      
      # gate contribution of the current state
      fh = self.linear_Fh(h)  # forget gate
      ih = self.linear_Ih(h)  # input gate
      zh = self.linear_Zh(h)  # cell update
      oh = self.linear_Oh(h)  # output gate

      fh = fh.view(bsz, self.n_heads, self.head_dim)
      ih = ih.view(bsz, self.n_heads, self.head_dim)
      zh = zh.view(bsz, self.n_heads, self.head_dim)
      oh = oh.view(bsz, self.n_heads, self.head_dim)
      check(fh, (bsz, self.n_heads, self.head_dim))

      f = torch.sigmoid(fx[:,:,idx,:] + fh)
      i = torch.sigmoid(ix[:,:,idx,:] + ih)
      z = torch.tanh(zx[:,:,idx,:] + zh)
      o = torch.sigmoid(ox[:,:,idx,:] + oh)
      check(f, (bsz, self.n_heads, self.head_dim))

      c = f * c + i * z
      h = o * torch.tanh(c)
      check(c, (bsz, self.n_heads, self.head_dim))
      check(h, (bsz, self.n_heads, self.head_dim))
      outputs.append(h)
    
    state = c, h

    # project all outputs
    outputs = torch.stack(outputs, dim=1)
    check(outputs, (bsz, self.seq_len, self.n_heads, self.head_dim))
    y = outputs.reshape(bsz, self.seq_len, self.n_heads * self.head_dim)
    if self.n_heads > 1:
      y = self.linear_post_head(y)
    check(y, (bsz, self.seq_len, self.h_dim))

    return y, state
  
  def __repr__(self):
    return f"MultiHeadLSTMCell(h_dim={self.h_dim}, head_dim={self.head_dim}, n_heads={self.n_heads}, name={self.name})"


class MultiHeadQuasiLSTMCell(nn.Module):
  """Implements an LSTM cell where every gate only depends on x and not on h for better parallelisation."""
  def __init__(self, seq_len, h_dim, head_dim, n_heads, name, block_length, max_seq_len=2048):
    super().__init__()
    self.name = name
    self.h_dim = h_dim
    self.head_dim = head_dim
    self.n_heads = n_heads
    self.block_length = block_length
    self.seq_len = seq_len

    # linear projections for x (done in parallel)
    self.linear_Fx = nn.Linear(h_dim, head_dim * n_heads, bias=True)
    torch.nn.init.normal_(self.linear_Fx.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(self.linear_Fx.bias)

    self.linear_Ix = nn.Linear(h_dim, head_dim * n_heads, bias=True)
    torch.nn.init.normal_(self.linear_Ix.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(self.linear_Ix.bias)

    self.linear_Zx = nn.Linear(h_dim, head_dim * n_heads, bias=True)
    torch.nn.init.normal_(self.linear_Zx.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(self.linear_Zx.bias)

    self.linear_Ox = nn.Linear(h_dim, head_dim * n_heads, bias=True)
    torch.nn.init.normal_(self.linear_Ox.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(self.linear_Ox.bias)

    # masks used to parallelise the quasi-lstm computation
    ones_mask = torch.tril(torch.ones((max_seq_len+1, max_seq_len)), diagonal=-1).T
    ones_mask = torch.reshape(ones_mask, (1,1,1,max_seq_len,max_seq_len+1))
    self.register_buffer("ones_mask", ones_mask)

    zeros_mask = torch.tril(torch.ones((max_seq_len+1, max_seq_len)), diagonal=-2).T
    zeros_mask = torch.reshape(zeros_mask, (1,1,1,max_seq_len,max_seq_len+1))
    self.register_buffer("zeros_mask", zeros_mask)

    # additional linear layer to project all heads back to h_dim in case of multiple heads (as in attention)
    self.linear_post_head = nn.Linear(head_dim * n_heads, h_dim, bias=True)
    torch.nn.init.normal_(self.linear_post_head.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(self.linear_post_head.bias)

  def forward(self, f_in, i_in, z_in, o_in, state, log=None):
    bsz, _, _ = f_in.shape
    
    c, h = state
    check(c, (bsz, self.n_heads, self.head_dim))
    check(h, (bsz, self.n_heads, self.head_dim))

    check(f_in, (bsz, self.seq_len, self.h_dim))
    check(i_in, (bsz, self.seq_len, self.h_dim))
    check(z_in, (bsz, self.seq_len, self.h_dim))
    check(o_in, (bsz, self.seq_len, self.h_dim))

    # compute the gates given x in parallel
    fx = self.linear_Fx(f_in)  # forget gate
    ix = self.linear_Ix(i_in)  # input gate
    zx = self.linear_Zx(z_in)  # cell update
    ox = self.linear_Ox(o_in)  # output gate

    # log activations
    log_stats_and_dist(fx, f"{self.name}.fx", log)
    log_stats_and_dist(ix, f"{self.name}.ix", log)
    log_stats_and_dist(zx, f"{self.name}.zx", log)
    log_stats_and_dist(ox, f"{self.name}.ox", log)

    # split sequence into blocks
    assert self.seq_len % self.block_length == 0
    n_blocks = self.seq_len // self.block_length

    fx = fx.view(bsz, n_blocks, self.block_length, self.n_heads, self.head_dim).permute(1,0,3,4,2)
    ix = ix.view(bsz, n_blocks, self.block_length, self.n_heads, self.head_dim).permute(1,0,3,4,2)
    zx = zx.view(bsz, n_blocks, self.block_length, self.n_heads, self.head_dim).permute(1,0,3,4,2)
    ox = ox.view(bsz, n_blocks, self.block_length, self.n_heads, self.head_dim).permute(1,0,3,4,2)
    check(fx, (n_blocks, bsz, self.n_heads, self.head_dim, self.block_length))
    
    f = torch.sigmoid(fx + 1.0)
    o = torch.sigmoid(ox)
    check(f, (n_blocks, bsz, self.n_heads, self.head_dim, self.block_length))

    update = (torch.sigmoid(ix) * torch.tanh(zx))
    check(update, (n_blocks, bsz, self.n_heads, self.head_dim, self.block_length))

    # below is the slow implementation
    """
    block_hs = []
    block_cs = []
    for block_idx in range(n_blocks):
      #curr_hs = []
      curr_cs = []
      for step_idx in range(self.block_length):
        c = c * f[block_idx,:,:,:,step_idx] + update[block_idx,:,:,:,step_idx]
        check(c, (bsz, self.n_heads, self.head_dim))
        curr_cs.append(c)

        #h = o[block_idx,:,:,:,step_idx] * torch.tanh(c)
        #check(h, (bsz, self.n_heads, self.head_dim))
        
        #curr_hs.append(h)
      
      cs = torch.stack(curr_cs, dim=-1)
      hs = o[block_idx] * torch.tanh(cs)
      check(hs, (bsz, self.n_heads, self.head_dim, self.block_length))
      #block_hs.append(torch.stack(curr_hs, dim=-1))
      block_hs.append(hs)
      #block_cs.append(cs)
    #"""

    # below is the identical fast implementation (less efficient but more parallel, thus more throughput)
    #"""
    c, h = state
    block_hs = []    
    for block_idx in range(n_blocks):
      curr_f = f[block_idx]
      check(curr_f, (bsz, self.n_heads, self.head_dim, self.block_length))

      # concatenate a 1.0 to the list of forget gates
      curr_f = torch.concatenate([
        curr_f, 
        torch.ones((bsz, self.n_heads, self.head_dim, 1), device=curr_f.device)
        ], dim=3)
      check(curr_f, (bsz, self.n_heads, self.head_dim, self.block_length+1))

      # tile curr_f to construct the matrix F
      F = torch.tile(curr_f.unsqueeze(3), (1, 1, 1, self.block_length, 1))
      check(F, (bsz, self.n_heads, self.head_dim, self.block_length, self.block_length+1))

      # mask upper triangle part with ones
      F = F.masked_fill(self.ones_mask[:,:,:,:self.block_length,:self.block_length+1] == 1.0, 1.0)

      # compute factorials and mask out the lower part again
      # uses cumsum because cumprod backward results in errors
      F = torch.cumsum(torch.log(F + 1e-8).flip(dims=[-1]), dim=-1).flip(dims=[-1]).exp() - 1e-8

      # mask upper triangle with zeroes
      F = F.masked_fill(self.zeros_mask[:,:,:,:self.block_length,:self.block_length+1] == 1.0, 0.0)

      # concatenate previous cell state with the cell updates
      check(c, (bsz, self.n_heads, self.head_dim))
      check(update, (n_blocks, bsz, self.n_heads, self.head_dim, self.block_length))
      u = torch.concatenate([c.unsqueeze(3), update[block_idx]], dim=3)
      
      check(u, (bsz, self.n_heads, self.head_dim, self.block_length+1))
      check(F, (bsz, self.n_heads, self.head_dim, self.block_length, self.block_length+1))
      cs = torch.einsum("bhdkl,bhdl->bhdk", F, u)
      check(cs, (bsz, self.n_heads, self.head_dim, self.block_length))

      # from the cell state at each step we compute the output
      hs = o[block_idx] * torch.tanh(cs)
      check(hs, (bsz, self.n_heads, self.head_dim, self.block_length))

      block_hs.append(hs)
      c = cs[:,:,:,-1]
    #"""

    hs = torch.stack(block_hs, dim=0)
    check(hs, (n_blocks, bsz, self.n_heads, self.head_dim, self.block_length))
    
    last_c = c
    last_h = block_hs[-1][:,:,:,-1]
    
    check(last_c, (bsz, self.n_heads, self.head_dim))
    check(last_h, (bsz, self.n_heads, self.head_dim))
    state = last_c, last_h

    # project all outputs
    y = hs.permute(1,0,4,2,3).reshape(bsz, self.seq_len, self.n_heads * self.head_dim)
    y = self.linear_post_head(y)
    check(y, (bsz, self.seq_len, self.h_dim))

    return y, state
  
  def __repr__(self):
    return f"MultiHeadQuasiLSTMCell(h_dim={self.h_dim}, head_dim={self.head_dim}, n_heads={self.n_heads}, block_length={self.block_length}, name={self.name})"


class MLP(torch.nn.Module):
    def __init__(self, h_dim, mlp_dim, n_layers, name):
        super().__init__()
        self.h_dim = h_dim
        self.mlp_dim = mlp_dim
        self.n_layers = n_layers
        self.name = name
        self.activation_fn = gelu
        
        self.c_fc = nn.Linear(h_dim, mlp_dim, bias=True)
        torch.nn.init.normal_(self.c_fc.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.c_fc.bias)
        
        self.c_proj = nn.Linear(mlp_dim, h_dim, bias=True)
        torch.nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02/math.sqrt(2 * self.n_layers))
        torch.nn.init.zeros_(self.c_proj.bias)
    
    def forward(self, x, log=None):
        bsz, seq_len, _ = x.shape
        
        check(x, (bsz, seq_len, self.h_dim))

        x = self.c_fc(x)
        check(x, (bsz, seq_len, self.mlp_dim))
        log_stats_and_dist(x, f"{self.name}.pre_act", log)

        x = self.activation_fn(x)
        log_stats_and_dist(x, f"{self.name}.post_act", log)
        
        x = self.c_proj(x)
        check(x, (bsz, seq_len, self.h_dim))
                
        return x
    
    def __repr__(self):
        return f"MLP(h_dim={self.h_dim}, mlp_dim={self.mlp_dim}, activation_fn={self.activation_fn}, name={self.name})"


class Block(nn.Module):
  def __init__(self, seq_len, h_dim, mlp_dim, head_dim, n_heads, n_layers, non_quasi, block_length, name):
    super().__init__()
    self.name = name
    self.h_dim = h_dim
    self.seq_len = seq_len
    self.non_quasi = non_quasi

    self.ln1 = LayerNorm(h_dim, name=f"{self.name}.ln1")
    if self.non_quasi:
      self.rnn = MultiHeadLSTMCell(seq_len=seq_len, h_dim=h_dim, head_dim=head_dim, n_heads=n_heads,
                                  name=f"{self.name}.MultiHeadLSTM")
    else:
      self.rnn = MultiHeadQuasiLSTMCell(seq_len=seq_len, h_dim=h_dim, head_dim=head_dim, n_heads=n_heads,
                                        block_length=block_length,
                                        name=f"{self.name}.MultiHeadQuasiLSTM")
                                  
    self.ln2 = LayerNorm(h_dim, name=f"{self.name}.ln2")
    self.mlp = MLP(h_dim=h_dim, mlp_dim=mlp_dim, n_layers=n_layers, name=f"{self.name}.MLP")

  def forward(self, x, state, log=None):
    bsz, _, _ = x.shape
    check(x, (bsz, self.seq_len, self.h_dim))

    ln_x = self.ln1(x, log=log)
    rnn_x, new_state = self.rnn(f_in=ln_x, i_in=ln_x, z_in=ln_x, o_in=ln_x, state=state, log=log)
    log_stats_and_dist(rnn_x, f"{self.name}.rnn_delta", log)
    x = x + rnn_x

    mlp_x = self.mlp(self.ln2(x, log=log), log=log)
    log_stats_and_dist(mlp_x, f"{self.name}.mlp_delta", log)
    x = x + mlp_x

    log_stats_and_dist(x, f"{self.name}.output", log)
    return x, new_state 