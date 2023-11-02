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

from torch import nn
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
    "use_flash": False,
}


class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        check_config(config, DEFAULT_CONFIG)
        self.c = c = config
        self.name = "GPT"
        
        self.input_embedding = nn.Embedding(c.vocab_size, c.h_dim)
        torch.nn.init.normal_(self.input_embedding.weight, mean=0.0, std=0.02)

        self.position_embedding = nn.Embedding(c.seq_len, c.h_dim)
        torch.nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.layers = nn.ModuleList([])
        for i in range(c.n_layers):
            self.layers.append(Block(h_dim=c.h_dim,
                                         mlp_dim=c.mlp_dim,
                                         head_dim=c.head_dim,
                                         n_heads=c.n_heads,
                                         n_layers=c.n_layers,
                                         name=f"{self.name}/Block{i+1}",
                                         use_flash=c.use_flash))
        
        self.ln_f = LayerNorm(c.h_dim, name=f"{self.name}/lnf")
        
        self.linear = nn.Linear(c.h_dim, c.vocab_size, bias=False)
        torch.nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)


    def get_init_state(self, batch_size, device):
       return None
    
    def forward(self, x, state, log=None):
        # x: [batch_size, seq_length]
        bsz, seqlen = x.shape
        c = self.c

        # embedd input tokens   
        x = self.input_embedding(x) * math.sqrt(c.h_dim)
        check(x, (bsz, seqlen, c.h_dim))

        # add position embedding
        pos_id = torch.arange(0, seqlen, dtype=torch.int64, device=c.device).unsqueeze(0)
        check(pos_id, (1, seqlen))
        pos = self.position_embedding(pos_id)
        check(pos, (1, seqlen, c.h_dim))
        x = x + pos

        # forward
        for layer in self.layers:
            x = layer(x, log=log)
            check(x, (bsz, seqlen, c.h_dim))
        
        # project to vocab
        x = self.ln_f(x, log=log)
        x = self.linear(x)
        check(x, (bsz, seqlen, c.vocab_size))
        
        return x, state


if __name__ == "__main__":
    # dummy config
    c = Munch()
    c.h_dim = 128
    c.mlp_dim = 256
    c.head_dim = 16
    c.n_heads = 8
    c.n_layers = 2

    c.batch_size = 8
    c.seq_len = 512
    c.vocab_size = 666
    c.use_flash = False
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