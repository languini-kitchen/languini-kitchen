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

"""Script to train models on the languini books dataset.

# Example calls:

## Single GPU:
CUDA_VISIBLE_DEVICES=0 torchrun --standalone languini/projects/lstm/main.py tiny --train_batch_size 16 --debug

## Multi GPU:
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=2 --master_addr=server.example.com --master_port=12300 \
    languini/projects/lstm/main.py tiny \
    --train_batch_size 16 \
    --eval_every 1000 \
    --log_grads_every 1000 \
    --tokens_per_second 36930 \
    --decay_steps 19475 \
    --max_train_steps 19475 \
    --gradient_accumulation_steps 2
"""


import os
import sys
import torch
import argparse
import torch.multiprocessing as mp

from languini.train_lib import lm_trainer
from languini.train_lib import lr_schedules
from languini.common_lib import parallel_utils
from languini.common_lib import experiment_utils
from languini.dataset_lib import languini_books

from languini.common_lib.parallel_utils import mprint
from languini.common_lib.parallel_utils import LOCAL_RANK, WORLD_RANK, WORLD_SIZE

import configs
from model import Model

def run(config, logger):
    c = config
    
    mprint(f"{c.n_workers} workers detected. Using DistributedDataParallel. Local rank: {LOCAL_RANK}")
    mprint(f"train batch size per worker/GPU: {c.train_batch_size // WORLD_SIZE}")
    mprint(f"eval batch size per worker/GPU: {c.eval_batch_size // WORLD_SIZE}")
    mprint(f"test batch size per worker/GPU: {c.test_batch_size // WORLD_SIZE}")
    mprint(f"gradient accumulation steps: {c.gradient_accumulation_steps}")

    mprint(f"WORLD_SIZE: {WORLD_SIZE}")  # total number of devices
    mprint(f"WORLD_RANK: {WORLD_RANK}")  # unique id within all devices
    mprint(f"LOCAL_RANK: {LOCAL_RANK}")  # unique id within the devices of this node

    mprint("Setup data sources ... ")
    # Compute the batch indices for this accelerator.
    assert c.train_batch_size % WORLD_SIZE == 0, "train batch size has to be a multiple of the number of workers"
    assert c.eval_batch_size % WORLD_SIZE == 0, "eval batch size has to be a multiple of the number of workers"
    train_batch_idxs = [i for i in range(c.train_batch_size) if i % WORLD_SIZE == WORLD_RANK]
    eval_batch_idxs = [i for i in range(c.eval_batch_size) if i % WORLD_SIZE == WORLD_RANK]

    END_OF_DOC_TOKEN = 2
    full_data_path = os.path.join(c.data_root, c.dataset)
    mprint(f"Loading data from {full_data_path}")
    train_ds = languini_books.LanguiniDatasetIterator(
        data_path=full_data_path,
        split='train',
        repeat=True,
        global_batch_size=c.train_batch_size,
        batch_idxs=train_batch_idxs,
        micro_batches=c.gradient_accumulation_steps,
        sequence_length=c.seq_len,
        device=c.device,
        end_of_doc_token=END_OF_DOC_TOKEN,
    )
    eval_ds = languini_books.LanguiniDatasetIterator(
        data_path=full_data_path,
        split='test',
        repeat=False,
        global_batch_size=c.eval_batch_size,
        batch_idxs=eval_batch_idxs,
        micro_batches=1,
        sequence_length=c.seq_len,
        device=c.device,
        end_of_doc_token=END_OF_DOC_TOKEN,
    )

    ## Setup Model
    mprint("Build model ... ")
    torch.manual_seed(c.seed)
    model = Model(config=c)
    if c.compile and c.compile != "None":
        model = torch.compile(model, mode=c.compile)
    model = model.to(c.device)
    if WORLD_SIZE > 1:
        mprint("running on multiple devices ...")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK])

    ## Setup Optimiser
    opt = torch.optim.Adam(model.parameters(), lr=c.max_lr, betas=(0.9, 0.95), eps=1e-08)
    scheduler = lr_schedules.CosineLR(opt,
                         warmup_steps=200,
                         max_lr=c.max_lr,
                         min_lr=c.min_lr,
                         max_steps=c.decay_steps,
                         decay_after=False)

    ## Setup Trainer
    trainer = lm_trainer.LMTrainer(config=c,
                        logger=logger,
                        model=model,
                        opt=opt,
                        scheduler=scheduler,
                        train_batches=train_ds,
                        eval_batches=eval_ds)

    ## Start Experiment
    mprint("Begin training ... ")    
    trainer.train()
    mprint("Done!")


def main():
    """Runs a Languini experiment using an LSTM model."""

    # initialise distributed processes
    device = parallel_utils.init_distributed()
    mp.set_start_method("spawn")

    mprint("Languini Experiment")

    # parse the config name
    config_name = experiment_utils.parse_config_name(configs.config_names)
    mprint(f"Loading config: {config_name}")

    # load the config file
    config = configs.load_config(name=config_name)
    project_path = os.path.dirname(os.path.abspath(__file__))
    mprint(f"project path: {project_path}")

    # create parser and add custom args not extracted from the config
    parser = experiment_utils.create_parser_based_on_config(config)
    parser.add_argument("--compile", default="default", type=str, help=f"Which compile mode to use (None, default, reduce-overhead, max-autotune)")

    # parse args and make updates to the config
    args = parser.parse_args(sys.argv[2:])
    config = experiment_utils.update_config_given_args(config, args)
    config.project_path = project_path
    config.device = device
    
    # Check if the config matches the available hardware
    config = experiment_utils.check_hardware(config, world_size=WORLD_SIZE)

    # Generate experiment name based on config
    configs.add_exp_name(config)
    mprint(f"experiment name: {config.exp_name}")
    
    # Create the log folder, backup python files, and backup the hyperparameter config to a file
    logger = experiment_utils.setup_experiment(config)
    
    run(config, logger)


if __name__ == "__main__":
    main()
