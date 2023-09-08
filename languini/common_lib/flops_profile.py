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
import sys
import pickle
import torch
import numpy as np

from importlib import import_module
from deepspeed.accelerator import get_accelerator
from deepspeed.profiling.flops_profiler import get_model_profile
from languini.common_lib import experiment_utils


def main():
    """Runs a throughput evaluations."""

    world_size = torch.cuda.device_count()
    if world_size != 1:
        print("Throughput measure should be done on a single accelerator.")
        exit(1)
    
    # Import the config and model and remove the corresponding args
    project_name = sys.argv[1]
    configs_module_path = f'languini.projects.{project_name}.configs'
    configs = import_module(configs_module_path)
    model_module_path = f'languini.projects.{project_name}.model'
    model_module = import_module(model_module_path)
    # Remove project name from the input arguments (needed for parse_config_name function)
    sys.argv = sys.argv[:1] + sys.argv[2:]

    # parse the config name
    config_name = experiment_utils.parse_config_name(configs.config_names)

    # load the config file
    config = configs.load_config(name=config_name)
    config.project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + f'/projects/{project_name}'

    # create parser and add custom args not extracted from the config
    parser = experiment_utils.create_parser_based_on_config(config)
    parser.add_argument("--compile", default="default", type=str, help=f"Which compile mode to use (None, default, reduce-overhead, max-autotune)")
    parser.add_argument("--results_pickle_file", default=None, type=str, help=f"Pickle file filename where to dump the results")

    # parse args and make updates to the config
    args = parser.parse_args(sys.argv[2:])
    if args.results_pickle_file:
        config = experiment_utils.update_config_given_args(config, args, verbose=False)
    else:
        print(f"Loading config: {config_name}")
        print(f"project path: {config.project_path}")
        config = experiment_utils.update_config_given_args(config, args, verbose=True)
    
    # Check if the config matches the available hardware
    config = experiment_utils.check_hardware(config, world_size=world_size)
    config.device = 'cuda'

    configs.add_exp_name(config)
    if not args.results_pickle_file:
        print(f"would be experiment name: {config.exp_name}")

    # create the pickle file for results (in case this function OOMs it will be empty, which we use as indicator of OOM)
    if args.results_pickle_file:
        open(config.results_pickle_file, 'w').close()

    # instantiate the model
    print_profile = False if args.results_pickle_file else True
    try:
        with get_accelerator().device(0):
            c = config
            model = model_module.Model(config=c).to(c.device)
            dummy_x = torch.tensor(np.random.choice(a=range(c.vocab_size), size=(c.train_batch_size, c.seq_len), replace=True)).to(c.device)
            state = model.get_init_state(batch_size=c.train_batch_size, device=c.device)
            flops, macs, params = get_model_profile(model=model,
                args=[dummy_x, state], # list of positional arguments to the model.
                kwargs=None, # dictionary of keyword arguments to the model.
                print_profile=print_profile, # prints the model graph with the measured profile attached to each module
                detailed=print_profile, # print the detailed profile
                module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                top_modules=1, # the number of top modules to print aggregated profile
                warm_up=10, # the number of warm-ups before measuring the time of each module
                as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                output_file=None, # path to the output file. If None, the profiler prints to stdout.
                ignore_modules=None) # the list of modules to ignore in the profiling
            
            if not args.results_pickle_file:
                print(f"{flops=}")
                print(f"{macs=}")
                print(f"{params=}")
    except torch.cuda.OutOfMemoryError:
        print("OutOfMemoryError")
        exit()

    pickle_dict = {
        'flops': flops,
        'macs': macs,
        'params': params,
    }

    if args.results_pickle_file:
        with open(config.results_pickle_file, 'wb') as f:
            pickle.dump(pickle_dict, f)

if __name__ == "__main__":
    main()
