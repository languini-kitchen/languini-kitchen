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
import math
import time
import shutil
import pickle
import tempfile

from csv import writer
from importlib import import_module
from languini.common_lib import experiment_utils


def main():
    # Import the config and model and remove the corresponding args
    project_name = sys.argv[1]
    configs_module_path = f'languini.projects.{project_name}.configs'
    configs = import_module(configs_module_path)
    # Remove project name from the input arguments (needed for parse_config_name function)
    sys.argv = sys.argv[:1] + sys.argv[2:]

    """Throughput evaluation of the current config. Stores results in csv file."""
    utils_path = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.dirname(utils_path) + f'/projects/{project_name}'
    csv_file = os.path.join(project_path, 'throughput_results.csv')
    tmp_folder = tempfile.mkdtemp()

    # parse the config name
    config_name = experiment_utils.parse_config_name(configs.config_names)
    print(f"Loading config: {config_name}")

    # load the config file
    config = configs.load_config(name=config_name)

    # create parser and add custom args not extracted from the config
    parser = experiment_utils.create_parser_based_on_config(config)
    parser.add_argument("--gpu", default=0, type=int, help="GPU on which to run the test")
    parser.add_argument("--compile", default="default", type=str, help=f"Which compile mode to use (None, default, reduce-overhead, max-autotune)")
    parser.add_argument("--batch_size_step", default=8, type=int, help=f"Initial batch size step to use when increasing batch size")

    # parse args and make updates to the config
    args = parser.parse_args(sys.argv[2:])
    config = experiment_utils.update_config_given_args(config, args, verbose=True)

    # Loop over all batch sizes
    curr_batch_size = 1
    min_batch_size = 1
    max_batch_size = math.inf
    batch_size_step = config.batch_size_step
    sweep_results = {}
    dummy_OOM_result = None
    first_round = True
    while batch_size_step != 0:
        print('+' * 50)
        print(f'Current batch size: {curr_batch_size}')
        
        throughput_file = f'{tmp_folder}/throughput_results_bsz{curr_batch_size}.pkl'
        flops_file = f'{tmp_folder}/flops_profile_results_bsz{curr_batch_size}.pkl'
       
        # Call throughput script
        command_args_string = ''
        for k, v in config.__dict__.items():
            if k == 'config_name' or k == 'train_batch_size' or k == 'batch_size_step' or k == 'gpu':
                continue
            if isinstance(v, bool):
                if v:
                    command_args_string += f' --{k}'
            else:
                command_args_string += f' --{k}={v}'

        command = f'CUDA_VISIBLE_DEVICES={config.gpu} python3 {utils_path}/throughput.py {project_name} {config_name} --results_pickle_file={throughput_file} --train_batch_size={curr_batch_size} {command_args_string}'
        print("\nEvaluating throughput ...")
        os.system(command)

        # Check if evaluation ran out of memory
        if os.path.getsize(throughput_file) == 0:
            # we need to reduce the step size if possible or exit the loop
            max_batch_size = curr_batch_size

            # Means this config OOMed so we store an empty results dict
            sweep_results[curr_batch_size] = {}
        else:
            # Throughput was successful, set new lowest batch size
            min_batch_size = curr_batch_size

            # load results dict
            with open(throughput_file, 'rb') as f:
                results_dict = pickle.load(f)
            results_dict['OOM'] = False

            # Call the flops script. 
            command = f'CUDA_VISIBLE_DEVICES={config.gpu} python3 {utils_path}/flops_profile.py {project_name} {config_name} --results_pickle_file={flops_file} --train_batch_size={curr_batch_size} {command_args_string}'
            print("\nEvaluating flops ... (can be out of memory before throughput)")
            os.system(command)

            # Evaluate flops result
            if os.path.getsize(flops_file) > 0:
                # Note: it can happen that the flops profile script fails, in which case we will not add the results to the sweep results
                # For some reason I've observed it to OOM even when the throughput script did not OOM
                # Get flops profile results
                with open(flops_file, 'rb') as f:
                    flops_profile_results = pickle.load(f)
                # Add flops profile results
                results_dict.update(flops_profile_results)
            sweep_results[curr_batch_size] = results_dict
            dummy_OOM_result = results_dict.copy()
        
        # Start with batch size 1 then keep increasing
        if first_round:
            curr_batch_size = batch_size_step
            first_round = False
        else:
            if min_batch_size + batch_size_step >= max_batch_size:
                batch_size_step = batch_size_step // 2
            
            curr_batch_size = min_batch_size + batch_size_step    
    print('Finished evaluation. Saving results ...')
    
    # Populate the entries that resulted in OOM with the dummy OOM result before writing to a CSV
    dummy_OOM_result['tokens per batch'] = 0
    dummy_OOM_result['tokens per second'] = 0
    for k, v in dummy_OOM_result.items():
        if isinstance(v, float):
            dummy_OOM_result[k] = 0
    dummy_OOM_result['OOM'] = True
    for k, v in sweep_results.items():
        if v == {}:
            sweep_results[k] = dummy_OOM_result
    
    # Sort the results
    sweep_results = dict(sorted(sweep_results.items()))

    # Write the CSV file header if it does not exist
    if not os.path.exists(csv_file):
        batch_sizes_evaluated = list(sweep_results.keys())
        with open(csv_file, 'w') as f:
            csv_writer = writer(f)
            csv_writer.writerow(['batch_size'] + list(sweep_results[batch_sizes_evaluated[0]].keys()))

    # Write the results to the CSV file
    with open(csv_file, 'a') as f:
        csv_writer = writer(f)
        for k, v in sweep_results.items():
            csv_writer.writerow([k] + list(v.values()))    
    print(f'Wrote the results to the CSV file {csv_file}.')

    # Delete the intermediate pickle files
    print(f'Removing intermediate pickle files...')
    shutil.rmtree(tmp_folder)
    print(f'Done.')

if __name__ == "__main__":
    main()
