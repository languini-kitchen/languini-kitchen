# GPT-2 component implementations based on the official TensorFlow code https://github.com/openai/gpt-2/blob/master/src/model.py

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
  

class CausalSelfAttention(nn.Module):
  def __init__(self, h_dim, head_dim, n_heads, name, use_flash, max_seq_len=4096):
    super().__init__()
    self.name = name
    self.h_dim = h_dim
    self.head_dim = head_dim
    self.n_heads = n_heads
    self.use_flash = use_flash
   
    self.linear_Q = nn.Linear(h_dim, head_dim * n_heads, bias=True)
    torch.nn.init.normal_(self.linear_Q.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(self.linear_Q.bias)

    self.linear_K = nn.Linear(h_dim, head_dim * n_heads, bias=True)
    torch.nn.init.normal_(self.linear_K.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(self.linear_K.bias)

    self.linear_V = nn.Linear(h_dim, head_dim * n_heads, bias=True)
    torch.nn.init.normal_(self.linear_V.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(self.linear_V.bias)

    self.linear_O = nn.Linear(head_dim * n_heads, h_dim, bias=True)
    torch.nn.init.normal_(self.linear_O.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(self.linear_O.bias)

    if not self.use_flash:
      # construct and store the following objects during initialisation for a faster forward pass
      self.dot_scale = float(head_dim) ** -0.5
      self.register_buffer("mask",
                          torch.tril(torch.ones(max_seq_len, max_seq_len))
                          .view(1, 1, max_seq_len, max_seq_len))
    
  def forward(self, query, key, value, log=None):
    bsz, seqlen, _ = query.shape
    check(query, (bsz, seqlen, self.h_dim))
    check(key, (bsz, seqlen, self.h_dim))
    check(value, (bsz, seqlen, self.h_dim))

    q = self.linear_Q(query)
    k = self.linear_K(key)
    v = self.linear_V(value)

    # log q,k,v activations
    log_stats_and_dist(q, f"{self.name}.q", log)
    log_stats_and_dist(k, f"{self.name}.k", log)
    log_stats_and_dist(v, f"{self.name}.v", log)

    q = q.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1,2)
    k = k.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1,2)
    v = v.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1,2)
    check(q, (bsz, self.n_heads, seqlen, self.head_dim))

    if self.use_flash:
       new_v = nn.functional.scaled_dot_product_attention(q, k, v, 
                                                          attn_mask=None, 
                                                          dropout_p=0.0, 
                                                          is_causal=True)
    else:
      # attn implementation
      dot = torch.einsum("bhid,bhjd->bhij", q, k) * self.dot_scale
      dot = dot.masked_fill(self.mask[:,:,:seqlen,:seqlen] == 0, float('-inf'))
      attn = F.softmax(dot, dim=-1)
      # the line below breaks when using torch compile with different eval batch size
      # furthermore, the line is significantly slower than the alternative
      # new_v = torch.einsum("bhjd,bhij->bhid", v, attn)
      # new_v = attn @ v  # also slower than below
      new_v = torch.einsum("bhij,bhjd->bhid", attn, v)
    
    check(new_v, (bsz, self.n_heads, seqlen, self.head_dim))
    new_v = new_v.transpose(1,2).reshape(bsz, seqlen, self.n_heads * self.head_dim)
    y = self.linear_O(new_v)
    return y
  
  def __repr__(self):
    return f"CausalSelfAttention(h_dim={self.h_dim}, head_dim={self.head_dim}, n_heads={self.n_heads}, name={self.name})"


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
  def __init__(self, h_dim, mlp_dim, head_dim, n_heads, n_layers, name, use_flash, max_seq_len=4096):
    super().__init__()
    self.name = name
    self.use_flash = use_flash
    self.h_dim = h_dim
    self.ln1 = LayerNorm(h_dim, name=f"{self.name}.ln1")
    self.attn = CausalSelfAttention(h_dim=h_dim, head_dim=head_dim, n_heads=n_heads,
                                    name=f"{self.name}.CausalAttn", 
                                    use_flash=self.use_flash,
                                    max_seq_len=max_seq_len)
    self.ln2 = LayerNorm(h_dim, name=f"{self.name}.ln2")
    self.mlp = MLP(h_dim=h_dim, mlp_dim=mlp_dim, n_layers=n_layers, name=f"{self.name}.MLP")

  def forward(self, x, log=None):
    bsz, seqlen, _ = x.shape
    check(x, (bsz, seqlen, self.h_dim))

    ln_x = self.ln1(x, log=log)
    attn_x = self.attn(query=ln_x, key=ln_x, value=ln_x, log=log)
    log_stats_and_dist(attn_x, f"{self.name}.attn_delta", log)
    x = x + attn_x

    mlp_x = self.mlp(self.ln2(x, log=log), log=log)
    log_stats_and_dist(mlp_x, f"{self.name}.mlp_delta", log)
    x = x + mlp_x

    log_stats_and_dist(x, f"{self.name}.output", log)
    return x 