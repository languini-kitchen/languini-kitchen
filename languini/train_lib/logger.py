# Copyright 2022 The Languini Authors.
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
import PIL
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt

from munch import Munch
from typing import Dict, List, Optional, Union
from torch.utils.tensorboard import SummaryWriter
from languini.common_lib import parallel_utils


class CustomLog:
    def to_tensorboard(self, name: str, summary_writer, global_step: int):
        pass

    def to_wandb(self):
        return None


class Scalar(CustomLog):
    def __init__(self, val: Union[torch.Tensor, np.ndarray, int, float]):
        if torch.is_tensor(val):
            val = val.item()

        self.val = val

    def to_tensorboard(self, name, summary_writer, global_step):
        summary_writer.add_scalar(name, self.val, global_step)

    def to_wandb(self):
        return self.val


class Scalars(CustomLog):
    def __init__(self, scalar_dict: Dict[str, Union[torch.Tensor, np.ndarray, int, float]]):
        self.values = {k: v.item() if torch.is_tensor(v) else v for k, v in scalar_dict.items()}
        self.leged = sorted(self.values.keys())

    def to_tensorboard(self, name, summary_writer, global_step):
        v = {k: v for k, v in self.values.items() if v == v}
        summary_writer.add_scalars(name, v, global_step)

    def to_wandb(self):
        return self.values


class TextTable(CustomLog):
    def __init__(self, header: List[str], data: List[List[str]]):
        self.header = header
        self.data = data

    def to_markdown(self):
        res = " | ".join(self.header)+"\n"
        res += " | ".join("---" for _ in self.header)+"\n"
        return res+"\n".join([" | ".join(l) for l in self.data])

    def to_tensorboard(self, name, summary_writer, global_step):
        summary_writer.add_text(name, self.to_markdown(), global_step)

    def to_wandb(self):
        return wandb.Table(data=self.data, columns=self.header)


class Logger:
    def __init__(self, config):
        c = config
        self.is_main =parallel_utils.is_main_process()

        if self.is_main:
            self.use_wandb = c.logger_type in ["all", "wandb"]
            self.use_tb = c.logger_type in ["all", "tb"]
            self.log_path = c.log_path
            self.wandb_run_dir = None

            if self.use_wandb:
                wandb_args = {
                    "project": c.wandb_project_name,
                    "config": Munch.toDict(c),
                }
                wandb.init(**wandb_args)
                self.wandb_run_dir = wandb.run.dir

            os.makedirs(self.log_path, exist_ok=True)
            if self.use_tb:
                tb_logdir = os.path.join(self.log_path, "tensorboard")
                os.makedirs(tb_logdir, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=tb_logdir, flush_secs=10)
            else:
                self.tb_writer = None

    def flatten_dict(self, dict_of_elems: Dict) -> Dict:
        res = {}
        for k, v in dict_of_elems.items():
            if isinstance(v, dict):
                v = self.flatten_dict(v)
                for k2, v2 in v.items():
                    res[k+"/"+k2] = v2
            else:
                res[k] = v
        return res

    def log(self, loglist: Union[List, Dict], step: Optional[int] = None):
        if not self.is_main:
            return
        
        if not isinstance(loglist, list):
            loglist = [loglist]

        loglist = [p for p in loglist if p]
        if not loglist:
            return

        d = {}
        for p in loglist:
            d.update(p)

        self.log_dict(d, step)

    def log_dict(self, dict_of_elems: Dict, step: Optional[int]):
        if not self.is_main:
            return
        
        dict_of_elems = self.flatten_dict(dict_of_elems)

        if not dict_of_elems:
            return

        dict_of_elems = {k: v.item() if torch.is_tensor(v) and v.nelement()==1 else v for k, v in dict_of_elems.items()}
        dict_of_elems = {k: Scalar(v) if isinstance(v, (int, float)) else v for k, v in dict_of_elems.items()}

        if self.use_wandb:
            wandbdict = {}
            for k, v in dict_of_elems.items():
                if isinstance(v, CustomLog):
                    v = v.to_wandb()
                    if v is None:
                        continue

                    if isinstance(v, dict):
                        for k2, v2 in v.items():
                            wandbdict[k+"/"+k2] = v2
                    else:
                        wandbdict[k] = v
                elif isinstance(v, plt.Figure):
                    wandbdict[k] = v
                else:
                    assert False, f"Invalid data type {type(v)}"

            wandbdict["step"] = step
            wandb.log(wandbdict)

        if self.tb_writer is not None:
            for k, v in dict_of_elems.items():
                if isinstance(v, CustomLog):
                    v.to_tensorboard(k, self.tb_writer, step)
                elif isinstance(v, plt.Figure):
                    self.tb_writer.add_figure(k, v, step)
                else:
                    assert False, f"Unsupported type {type(v)} for entry {k}"
    
    def log_text(self, log_key: str, log_text: str, step: Optional[int] = None):
        if not self.is_main:
            return
        
        if self.use_wandb:
            # wandb.log({log_key: wandb.Html(text)}, step=step)
            self.log({log_key: TextTable([log_key], [[log_text]])}, step=step)
        
        if self.tb_writer is not None:
            self.tb_writer.add_text(log_key, log_text, step)
