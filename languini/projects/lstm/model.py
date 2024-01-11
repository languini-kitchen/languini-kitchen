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
import numpy as np 

from torch import nn, Tensor
from munch import Munch
from languini.train_lib.train_utils import check_config
from languini.common_lib.debug_utils import check
from languini.common_lib.debug_utils import log_stats_and_dist

from lib import LayerNorm
from lib import Block


DEFAULT_CONFIG = {
    "device": None,
    "vocab_size": None,
    "n_layers": None,
    "h_dim": None,
    "mlp_dim": None,
    "head_dim": None,
    "n_heads": None,
    "non_quasi": None,
    "block_length": None,
}


class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        check_config(config, DEFAULT_CONFIG)
        self.c = c = config
        self.name = "LSTM"
        
        self.input_embedding = nn.Embedding(c.vocab_size, c.h_dim)
        torch.nn.init.normal_(self.input_embedding.weight, mean=0.0, std=0.02)

        #self.position_embedding = nn.Embedding(c.seq_len, c.h_dim)
        #torch.nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.layers = nn.ModuleList([])
        for i in range(c.n_layers):
            self.layers.append(Block(seq_len=c.seq_len,
                                     h_dim=c.h_dim,
                                     mlp_dim=c.mlp_dim,
                                     head_dim=c.head_dim,
                                     n_heads=c.n_heads,
                                     n_layers=c.n_layers,
                                     non_quasi=c.non_quasi,
                                     block_length=c.block_length,
                                     name=f"{self.name}/Block{i+1}"))
        
        self.ln_f = LayerNorm(c.h_dim, name=f"{self.name}/lnf")
        
        self.linear = nn.Linear(c.h_dim, c.vocab_size, bias=False)
        torch.nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)

    def get_init_state(self, batch_size, device):
        return [
            (torch.zeros((batch_size, self.c.n_heads, self.c.head_dim), device=device),
             torch.zeros((batch_size, self.c.n_heads, self.c.head_dim), device=device))
            for _ in range(self.c.n_layers)
        ]

    
    def forward(self, x, state, log=None):
        # x: [batch_size, seq_length]
        bsz, seqlen = x.shape
        c = self.c

        # embedd input tokens   
        x = self.input_embedding(x) * math.sqrt(c.h_dim)
        check(x, (bsz, seqlen, c.h_dim))

        # forward
        new_states = []
        for idx, layer in enumerate(self.layers):
            check(state[idx][0], (bsz, self.c.n_heads, self.c.head_dim))
            check(state[idx][1], (bsz, self.c.n_heads, self.c.head_dim))

            x, new_state = layer(x, state[idx], log=log)

            check(x, (bsz, seqlen, c.h_dim))
            check(new_state[0], (bsz, self.c.n_heads, self.c.head_dim))
            check(new_state[1], (bsz, self.c.n_heads, self.c.head_dim))

            new_states.append(new_state)
        
        # project to vocab
        x = self.ln_f(x, log=log)
        x = self.linear(x)
        check(x, (bsz, seqlen, c.vocab_size))
        
        return x, new_states


if __name__ == "__main__":
    # dummy config
    c = Munch()
    c.h_dim = 128
    c.mlp_dim = 256
    c.head_dim = 16
    c.n_heads = 8
    c.n_layers = 2

    c.batch_size = 8
    c.block_length = 32
    c.seq_len = 512
    c.vocab_size = 666
    c.non_quasi = False
    if torch.cuda.is_available():
        c.device = 'cuda'
        c.n_workers = torch.cuda.device_count()
    else:
        c.device = "cpu"
        c.n_workers = 1
    c.device_batch_size = c.batch_size // c.n_workers

    model = Model(config=c).to(c.device)

    dummy_x = torch.tensor(np.random.choice(a=range(c.vocab_size), size=(c.batch_size, c.seq_len), replace=True))
    dummy_x = dummy_x.to(c.device)
    print(f"{dummy_x.shape=}")

    state = model.get_init_state(batch_size=c.batch_size, device=c.device)
    logits, state = model(dummy_x, state)
    print(f"{logits.shape=}")

    print(model)