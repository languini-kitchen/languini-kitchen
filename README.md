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
```
git clone https://github.com/languini-kitchen/languini-kitchen.git
cd languini-kitchen

python3 -m venv venv
source venv/bin/activate

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

Look up further details in the respective project folders.
