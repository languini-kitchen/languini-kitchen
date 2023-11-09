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

from importlib import import_module
from languini.common_lib import experiment_utils
from languini.common_lib import throughput_utils


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
        open(args.results_pickle_file, 'w').close()

    # instantiate the model
    model = model_module.Model(config=config)
    if config.compile != "None":
        model = torch.compile(model, mode=config.compile)
    
    # measure the speed of the forward and backward pass of the model
    try:
        result = throughput_utils.throughput_test(config=config, model=model)
    except torch.cuda.OutOfMemoryError:
        print("OutOfMemoryError")
        exit()

    # present results to terminal
    if not args.results_pickle_file:
        print("\nResults:")
        del result['config']
        for k,v in result.items():
            if isinstance(v, int):
                print(f"{k}: {v:,}")
            else:
                print(f"{k}: {v}")

        tokens_per_second = result["tokens per second"]
        tokens_per_batch = result["tokens per batch"]
        
        print(f"\nOn this device, the measured throughput permits the following number of training steps.\n")
        hours = [3, 6, 12, 24, 48, 98, 192, 384]
        for h in hours:
            total_tokens = tokens_per_second * 60 * 60 * h
            steps = total_tokens // tokens_per_batch
            print(f"{h:3d} hours: {total_tokens:15,} tokens in {steps:10,} steps")

    # Add model config information to the results dict (for logging purposes) and dump to pickle file
    pickle_dict = {}
    pickle_dict['vocab_size'] = config.vocab_size
    pickle_dict['config_name'] = config_name
    pickle_dict['seq_len'] = config.seq_len
    pickle_dict['n_layers'] = config.n_layers
    pickle_dict['h_dim'] = config.h_dim
    pickle_dict['mlp_dim'] = config.mlp_dim
    pickle_dict['head_dim'] = config.head_dim
    pickle_dict['n_heads'] = config.n_heads
    pickle_dict['compile'] = config.compile
    #pickle_dict['use_flash'] = config.use_flash
    pickle_dict.update(result)

    if args.results_pickle_file:
        with open(args.results_pickle_file, 'wb') as f:
            pickle.dump(pickle_dict, f)

if __name__ == "__main__":
    # The commented lines would trace out the computation and allow you to use pytorch's profiler.
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    #torch.cuda.synchronize()
    #profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True)
    #profiler.__enter__()

    main()

    #torch.cuda.synchronize()
    #profiler.__exit__(None, None, None)
    #profiler.export_chrome_trace("trace_all.json")
