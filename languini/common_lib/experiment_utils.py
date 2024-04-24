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
import torch
import shutil
import pickle
import zipfile
import argparse
import re
import wandb
from languini.train_lib import logger
from languini.common_lib.parallel_utils import mprint
from languini.common_lib.parallel_utils import is_main_process


def check_hardware(config, world_size):
    config.n_workers = world_size
    assert config.train_batch_size % config.n_workers == 0, \
            f"Ensure that train_batch_size ({config.train_batch_size}) is a multiple of the number of GPUs available ({config.n_workers})."
    assert config.eval_batch_size % config.n_workers == 0, \
            f"Ensure that eval_batch_size ({config.eval_batch_size}) is a multiple of the number of GPUs available ({config.n_workers})."
    assert config.test_batch_size % config.n_workers == 0, \
            f"Ensure that test_batch_size ({config.test_batch_size}) is a multiple of the number of GPUs available ({config.n_workers})."
    return config

def setup_log_folder(log_path):
  """Creates a log folder within the project folder. """
  if not os.path.exists(log_path):
      print(f"creating new log directory: {log_path}")
      os.makedirs(log_path)
      return
  
  # folder already exists, ask how to proceed.
  print("\nWARNING: The results directory already exists:\n{}".format(log_path))
  print("Delete previous results directory [y/n]? ", end="")
  choice = input()
  while choice not in ["y", "Y", "n", "N"]:
      print("invalid answer. try again.", end="")
      choice = input()

  if choice == "y" or choice == "Y":
      print("removing old directory ...")
      shutil.rmtree(log_path)
      print("creating new log directory...")
      os.makedirs(log_path)
  else:
      print("Exiting.")
      sys.exit()


def save_source_code(project_path, logger):
    """Takes all python files of the languini folder, zips them, and stores them. """

    # recursively find all all python files in the languini folder
    languini_dir_path = os.path.join(project_path, "../..")
    python_files = []
    for dir_path, _, file_names in os.walk(languini_dir_path):
        for file_name in file_names:
            if file_name.endswith(".py"):
                python_files.append((os.path.join(dir_path, file_name), file_name))

    # zip the files into the log folder
    def save_zipfile(path):
        with zipfile.ZipFile(path, mode="w") as zf:
            for file_path, file_name in python_files:
                rel_path = os.path.relpath(file_path)
                with open(file_path, "r") as f:
                    zf.writestr(rel_path, f.read())
        print(f"Successfully created a source code backup for {len(python_files)} python files.")

    if logger.use_tb:
        zip_file_path = os.path.join(logger.log_path, "source.zip")
        save_zipfile(zip_file_path)
    if logger.use_wandb:
        zip_file_path = os.path.join(logger.wandb_run_dir, "source.zip")
        save_zipfile(zip_file_path)


def setup_experiment(config):
    config.log_path = os.path.join(config.project_path, config.relative_log_path, config.exp_name)
    
    if not is_main_process():
        return logger.Logger(config)

    # create a log folder and warn if it already exists
    setup_log_folder(config.log_path)

    # create logger
    logger_obj = logger.Logger(config)
    
    # save the current python files as zip file in the log folder
    save_source_code(config.project_path, logger_obj)

    # save the used config in the log folder
    logger_obj.save_file(config, "config.pickle")

    return logger_obj


def parse_config_name(config_names):
    # parser for the config name
    valid_config_names = '\n'.join([f'\t{name}' for name in config_names])
    config_parser = argparse.ArgumentParser(description='Runs an experiment given a specific config.', usage=f"main.py <config> [<args>] \n\nValid config names are:\n{valid_config_names}")
    config_parser.add_argument('config', help='Name of the config (default: \'default\')', default='default', nargs='?')
    
    # only parse the first argument
    config_args = config_parser.parse_args(sys.argv[1:2])
    return config_args.config


def create_parser_based_on_config(config):
    """Creates an ArgumentParser based on the entries of the config."""
    parser = argparse.ArgumentParser()
    for key, value in config.items():
        if type(value) == bool and value:
            parser.add_argument(f"--{key}", action='store_false', help='Boolean flag. Use without args to toggle. Default: True')
        elif type(value) == bool and not value:
            parser.add_argument(f"--{key}", action='store_true', help='Boolean flag. Use without args to toggle. Default: False')
        else:
            parser.add_argument(f"--{key}", default=value, type=type(value), help=f"default: {value}")
    return parser


def update_config_given_args(config, args, verbose=True):
    """Make changes to the config based on command line arguments."""
    if verbose:
        mprint("Arguments:")
    for key, value in vars(args).items():
        if verbose:
            mprint(f"\t{key}={value}")
        if not key in config:
            config[key] = value
        elif config[key] != value:
            if verbose:
                mprint(f"\toverwriting {key} to {value}")
            config[key] = value
    return config


def load_wandb_files(run_path, exclude=None, include=None):
    """
    Downloads all files of a wandb run into a cache directory if not cached already

    Args:
        run_path: The wandb run path
        exclude: A list of regex patterns to exclude from download (default is none)
        include: A list of regex patterns to include in download (default is all)
    
    Returns:
        The path to the directory with the run's files
    """
    api = wandb.Api()
    run = api.run(run_path)
    files = run.files()

    run_dir = os.path.join(os.getcwd(), "cache", run_path.replace("/", "_"))
    os.makedirs(run_dir, exist_ok=True)

    for file in files:
        if os.path.exists(os.path.join(run_dir, file.name)):
            continue
        # check if file.name matches any regex
        if exclude is not None and any([re.match(pattern, file.name) for pattern in exclude]):
            continue
        if include is not None and not any([re.match(pattern, file.name) for pattern in include]):
            continue
        
        print(f"Downloading file {file.name} from run {run_path}...")
        file.download(root=run_dir, exist_ok=False)
    
    return run_dir 


def load_wandb_checkpoint_and_config(run_path):
    """
    Load a checkpoint and config from a wandb run path.
    """
    wandb_dir = load_wandb_files(run_path, include=["config.pickle", ".*model.pt"])
    checkpoint_file = os.path.join(wandb_dir, "checkpoints/model.pt")
    config_file = os.path.join(wandb_dir, "config.pickle")
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Could not load checkpoitn file from wandb run {run_path}")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Could not load config file from wandb run {run_path}")
    return checkpoint_file, config_file

