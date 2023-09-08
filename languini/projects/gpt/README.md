# GPT Baseline Model

This model is based on the official GPT-2 TensorFlow model with learned position encodings. 

Run the following command to see if you can load the model and feed it a batch of dummy data.
```
CUDA_VISIBLE_DEVICES=0 python3 languini/projects/gpt/model.py
```

Example command to train on one machine with a single GPU.
```
torchrun --standalone languini/projects/gpt/main.py mini --log_terminal_every 10
```

Example command to train on one machine with multiple GPUS.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=4 --master_addr=myserver.example.ch --master_port=12300 languini/projects/gpt/main.py
```

## Train the best config
The following configs achieve the best results for the 6h, 12h, 24h, 48h, and 96h compute scales. You can compute the number of steps given the compute class and the batch sizes with the load_throughput.ipynb notebook.

```tokens per second``` is only needed to produce a plot of performance over accelerator hours which allows comparisons during training. Recall that throughput is the theoretically largest throughput achieved on our reference hardware and simulates ideal load. During training, this doesn't have to be case. Furthermore, during training we can also train on multiple machines or on a different hardware. But the theoretical throughput measure must remain the same. See the Languini Kitchen paper for more details. 

6h: GPT small, bsz 128, seqlen 512 -> 55416 tokens per second, 18265 steps, 4 gradient accumulation steps
```
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=localhost --master_port=12300 languini/projects/gpt/main.py small \
  --train_batch_size 128 \
  --decay_steps 18265 \
  --max_train_steps 18265 \
  --gradient_accumulation_steps 4 \
  --tokens_per_second 55416 \
  --log_terminal_every 100 \
  --eval_every 500 \
  --log_grads_every 1000 \
  --log_ckpt_every 1000
```

12h: GPT small, bsz 128, seqlen 512 -> 55416 tokens per second, 36529 steps, 4 gradient accumulation steps
```
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=localhost --master_port=12300 languini/projects/gpt/main.py small \
  --train_batch_size 128 \
  --decay_steps 36529 \
  --max_train_steps 36529 \
  --gradient_accumulation_steps 4 \
  --tokens_per_second 55416 \
  --log_terminal_every 100 \
  --eval_every 500 \
  --log_grads_every 1000 \
  --log_ckpt_every 1000
```

24h: GPT small, bsz 256, seqlen 512 -> 55416 tokens per second, 36529 steps, 8 gradient accumulation steps
```
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=localhost --master_port=12300 languini/projects/gpt/main.py small \
  --train_batch_size 256 \
  --decay_steps 36529 \
  --max_train_steps 36529 \
  --gradient_accumulation_steps 8 \
  --tokens_per_second 55416 \
  --log_terminal_every 100 \
  --eval_every 500 \
  --log_grads_every 1000 \
  --log_ckpt_every 1000
```

48h: GPT medium, bsz 256, seqlen 512 -> 16618 tokens per second, 21908 steps, 32 gradient accumulation steps
```
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=localhost --master_port=12300 languini/projects/gpt/main.py medium \
  --train_batch_size 256 \
  --decay_steps 21908 \
  --max_train_steps 21908 \
  --gradient_accumulation_steps 32 \
  --tokens_per_second 16618 \
  --log_terminal_every 100 \
  --eval_every 500 \
  --log_grads_every 1000 \
  --log_ckpt_every 1000
```

96h: GPT medium, bsz 256, seqlen 512 -> 16618 tokens per second, 43817 steps, 32 gradient accumulation steps
```
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=localhost --master_port=12300 languini/projects/gpt/main.py medium \
  --train_batch_size 256 \
  --decay_steps 43817 \
  --max_train_steps 43817 \
  --gradient_accumulation_steps 32 \
  --tokens_per_second 16618 \
  --log_terminal_every 100 \
  --eval_every 500 \
  --log_grads_every 1000 \
  --log_ckpt_every 1000
```

## Evaluate
```
P=languini/projects/gpt/logs/24h/large/GPT_books16384_bsz128_sl512_coslr0.0006to6e-06_h768_ff3072_nH12_dH64_nl12_clip0.0_decay36k_gpus1_defaultCompile_fp16_seed0
CUDA_VISIBLE_DEVICES=0 torchrun --standalone languini/projects/gpt/eval.py \
        --checkpoint_file "$P"/checkpoints/model.pt \
        --config_file "$P"/config.pickle \
        --eval_data_split test \
        --last_n 128
```