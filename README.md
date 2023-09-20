<div align="center">
  
# Languini Kitchen

[![Website](https://img.shields.io/badge/Website-Visit-brightgreen)](https://languini-kitchen.github.io/) 
[![ArXiv](https://img.shields.io/badge/ArXiv-Paper-blue)](https://arxiv.org/) 
[![Discord](https://img.shields.io/badge/Discord-Join-purple)](https://discord.gg/5W3rTRJa) 
[![Twitter](https://img.shields.io/badge/Twitter-Follow-blue?logo=twitter&style=social)]([#](https://twitter.com/LanguiniKitchen)https://twitter.com/LanguiniKitchen)

Enabling Language Modelling Research at Different Scales of Compute.

</div

Languini is designed to be a research codebase for the development of small language models. The code is easy to use, simple to understand, and hackable.

## Preparations
Download books3, tokenise Languini books, and get the Languini codebase ready for experiments.

### Install Languini
Note: torch.compile is not yet supported in python3.11. You will have to install an older version of python in that case before continuing.

```
git clone https://github.com/languini-kitchen/languini-kitchen.git
cd languini-kitchen
```

```
python3 -m venv venv
source venv/bin/activate
```

```
pip install --upgrade pip setuptools
pip install -e . --upgrade
```

### Download and tokenise the books3 dataset

```
chmod +x languini/dataset_lib/easy_download_and_tokenise.sh
./languini/dataset_lib/easy_download_and_tokenise.sh
```

## How to run experiments
Use the following command to train a small transformer model on languini books. 
```
torchrun --standalone languini/projects/gpt/main.py
```

The baseline main files consist of two arg parsers. The first argument is the project name. Given the project name, a second argparser is created based on all entries in the configs.py file. This allows us to easy modify the any hyperparameter that is listed in configs.py.

```
torchrun --standalone languini/projects/gpt/main.py tiny --h_dim 666
```

If you have multiple GPUs available you need to specify the number of GPUs and master server. Training across different machines (nodes) is only recommended if your network is fast enough. For example, use the following command to run on GPU with ids 0 and 2. 
```
CUDA_VISIBLE_DEVICES=0,2 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=2
    --master_addr=localhost --master_port=12303 languini/projects/gpt/main.py mini \
    --max_train_steps 50000 \
    --decay_steps 50000 \
    --logger_type tb \
    --train_batch_size 32 \
    --max_eval_steps 200 \
    --gradient_accumulation_steps 1 \
    --eval_every 100 \
    --log_terminal_every 20 \
    --log_metrics_every 20 \
    --log_grads_every 100
```

| Argument | Description |
| --- | --- |
| CUDA_VISIBLE_DEVICES=0,2 | Only exposes gpu device 0 and 2 to pytorch
| torchrun | PyTorch tool to start distributed scripts (we always use torchrun)
| nnodes | Number of nodes/machines in total
| node_rank | Unique rank of this node/machine; rank 0 is the master
| nproc_per_node | Number of workers per node, each worker will use one gpu
| master_addr | master server which performs the weight updates
| master_port | master port

The remaining arguments are specific to the projects ```config.py```.

## Measure throughput
Use the following command to measure throughput and flops of any model config.
```
CUDA_VISIBLE_DEVICES=0 python3 languini/common_lib/flops_profile.py gpt small --train_batch_size 1
```

```
CUDA_VISIBLE_DEVICES=0 python3 languini/common_lib/throughput.py gpt small --train_batch_size 8
```

Use the following command to automatically find the largest batch size for a particular model and save all throughput results in a csv file of the respective project folder.
```
python3 languini/common_lib/throughput_sweep.py gpt tiny --gpu 0
```

Look up further details in the respective project folders.
