<div align="center">
  
# Languini Kitchen

[![Website](https://img.shields.io/badge/Website-Visit-brightgreen)](https://languini-kitchen.github.io/) 
[![ArXiv](https://img.shields.io/badge/ArXiv-Paper-blue)](https://arxiv.org/abs/2309.11197) 
[![Discord](https://img.shields.io/badge/Discord-Join-purple)](https://discord.gg/5W3rTRJa) 
[![Twitter](https://img.shields.io/badge/Twitter-Follow-blue?logo=twitter&style=social)]([#](https://twitter.com/LanguiniKitchen)https://twitter.com/LanguiniKitchen)

Enabling Language Modelling Research at Different Scales of Compute.

<img src="robot.webp" alt="An image of a robot in a lagnuini kitchen." width="350"/>
</div>

[Introduction](README.md#introduction) <br>
[Download](README.md#download) <br>
\- [Install Languini](README.md#install-languini) <br>
\- [Download and tokenise the books3 dataset](README.md#download-and-tokenise-the-books3-dataset) <br>
[How to run experiments](README.md#how-to-run-experiments) <br>
[How to evaluate models on the languini books benchmark](README.md#how-to-evaluate-models-on-the-languini-books-benchmark) <br>
[Step by step instructions for research on languini](README.md#step-by-step-instructions-for-research-on-languini) <br>
[Current Model Implementation](README.md#current-model-implementation) <br>
[Frequently Asked Questions](README.md#frequently-asked-questions) <br>
[Changelog](README.md#changelog)

## Introduction

Languini is designed to be a research codebase for the development of language models. The code is easy to use, simple to understand, and hackable. Follow the instructions in the following Sections to learn how to install the languini pip package, how to run your own experiments, and how to evaluate your research on the languini books benchmark. If you have questions you should be able to find help on our discord channel or you can reach out to the authors of the Languini Kitchen paper. All relevant links can be found at the top of this readme.

## Download
Here we will download books3, tokenise Languini books, and get the Languini codebase ready for experiments.

### Install Languini
Note: torch.compile is not yet supported in Python 3.11. You will have to install an older version of Python in that case before continuing. If you'd like to copy languini into your own private repo for your research follow [these steps](create_private_repo.md).

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

This process is slow and can take a while.
```
./languini/dataset_lib/easy_download_and_tokenise.sh
```

If you already downloaded the tokenised data you can just link to it.
```
ln -s path/to/data data
```

Check dataset integrity by hashing filenames and filesizes. 
```
python3 languini/dataset_lib/integrity_test.py data/books/books_16384
```
The hash for ```books_16384``` is ```0957d626c108d0075c18a5a99fa766533a46b2fba833061fdce445c6066558f7```.


## How to run experiments
Use the following command to train a small transformer model on languini books. 
```
torchrun --standalone languini/projects/gpt/main.py
```

The baseline main files consist of two argument parsers. The first argument is the project name. Given the project name, a second argument parser is automatically created based on all entries of the project's configs.py file. This allows us to easily modify any hyperparameter that is listed in configs.py.

```
torchrun --standalone languini/projects/gpt/main.py tiny --h_dim 666
```

If you have multiple GPUs available you need to specify the number of GPUs and master server. For example, use the following command to run on GPU with IDs 0 and 2. Training across different machines (nodes) is only recommended if your network is fast enough. 
```
CUDA_VISIBLE_DEVICES=0,2 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=2 \
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

Running various experiments on different GPUs and different servers is impractical with config edits. However, using arguments can also become cumbersome. To simplify this we recommend the use of two simple scripts. First, you can write in ```run.sh``` your custom command for your next experiment. Second, you can use ```sync.sh``` to easily sync your languini code to a specific server. All that is left to do is to connect to the server and execute e.g. ```./run.sh 0,3,4,5``` to run your commands on 4 GPUs.


| Argument | Description |
| --- | --- |
| CUDA_VISIBLE_DEVICES=0,2 | Only exposes GPU device 0 and 2 to PyTorch
| torchrun | PyTorch tool to start distributed scripts (we always use torchrun)
| nnodes | Number of nodes/machines in total
| node_rank | Unique rank of this node/machine; rank 0 is the master
| nproc_per_node | Number of workers per node, each worker will use one GPU
| master_addr | master server which performs the weight updates
| master_port | master port

The remaining arguments are specific to the projects ```config.py```.


## How to evaluate models on the languini books benchmark

We compare models on the languini books benchmark based on the *normalised perplexity* on held-out data (see the paper for more details on normalised perplexity). 
Because larger models tend to perform better, all models are compared given the same compute constraints. Before training, we use the provided throughput script (see below) to measure the tokens per second of a particular model and config on the reference hardware. Given the throughput of a config we can then compute the number of training steps for different hours of compute. The ```languini/dataset_lib/throughput.py``` script will directly output the number of steps for the compute hours used in the paper. For further details please see Section 3 of the Languini Kitchen paper.

Use the following command to measure the throughput and flops of any model config.
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

## Step by step instructions for research on languini
1. Follow the languini-kitchen instructions to install the package and to download and tokenise the training data.
2. Create a new project folder (we'd recommend to copy one of the existing ones) and give it a distinct name. 
3. Implement your method only within that project folder. If you want to use code from other projects, copy that code into your own project.
4. Use the ```throughput.py``` and ```throughput_sweep.py``` scripts to improve your models throughput. It is recommended to max out GPU utilization on the reference hardware.
5. Use your best config and ```throughput.py``` to calculate the number of steps for different hours of compute.
6. Train the same model and config on all compute classes that you can afford for the respective number of steps. You can track the experiments using wandb or tensorboard.
7. Evaluate your final model using your project's ```eval.py``` script on all test splits. You can compare absolute performance at different levels of compute or you can also compare the scaling trend of your model.

If you would like to share your work we recommend these additional steps.

8. Write a detailed report of your method and findings. Publish it on arXiv or similar.
9. Upload your final model checkpoints and logs to zenodo.org.
10. Add a readme to your project with commands to reproduce your results, download links for your logs and trained models, and a BibTeX to your report.
11. Create a push-request on the official languini-kitchen repository. 

Successfully pushing your project to the official languini-kitchen repository will make it easier for others to see your work and build on it. New projects and results may be also mentioned on any of Languini's social media channels.  

## Current Model Implementation
These are the current models which are implemented within languini. You can find the commands to reproduce their respective results in the project's readme. 

| Model | Project | Brief Description |
| :--: | :--: | :--: |
| [GPT](languini/projects/gpt/) | Initial Baselines | GPT-2 inspired transformer baseline | 
| [LSTM](languini/projects/lstm/) | Initial Baselines | Classic LSTM |  
| [Quasi-LSTM](languini/projects/lstm/) | Initial Baselines | Quasi LSTM with block-parallel implementation | 


## Frequently Asked Questions

___Q: What does the ```train_batch_size``` argument describe?___

The ```train_batch_size``` argument describes the number of sequences used for each step of the optimiser. This is a much more meaningful number than the size of a batch per accelerator or per gradient accumulation step. Internally, the ```train_batch_size``` argument is thus divided by the number of accelerators and gradient accumulation steps which results in the (local) batch size of an accelerator for each forward pass.

___Q: Is the training on 8 accelerators equivalent to training on 1 accelerator with 8 gradient accumulation steps?___

Yes, it is ***identical***. This required some changes in our custom dataloader and is probably best explained in an example. If your ```train_batch_size``` is e.g. 80, the dataloader will produce 80 parallel sequence streams. If you train on 8 GPUs, each GPU will process its exclusive 10 sequence streams in parallel and make updates to its exclusive 10 model states (in case the model is stateful). If you train on 1 GPU with 8 gradient accumulation steps, the same computation is performed sequentially.

___Q: How does the dataloader load the books?___

The dataloader creates ```train_batch_size``` sequence streams. The books are loaded in a deterministic order and loaded into the shortest stream with the lowest index. Each accelerator has its own dataloader which only loads its exclusive data into GPU memory.

___Q: Why does the throughput print during training not match the published throughput results?___

Throughput should be measured ***before*** the training begins with the `languini/common_lib/throughput.py` and `languini/common_lib/throughput_sweep.py` scripts. These scripts feed random indices to the model and measure the average time it takes to perform a forward pass, backward pass, and weight update. This allows the researcher to calculate the number of tokens they can process for specific levels of compute and also allows them to train the model on hardware that is different from the reference hardware. The throughput logs created during training can vary due to the hardware, the load of the entire system, or other potential bottlenecks like memory transfer, disk reading speed, or network speed (in the case of distributed training).

## Changelog

### v0.0.2

Fix RNN state-carrying bug for RNN trained with gradient accumulation. 

### v0.0.1

Initial release.
