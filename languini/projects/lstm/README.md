# Quasi-LSTM Baseline Model

This is our LSTM (use --non_quasi) and quasi LSTM implementation (default). An RNN that stays as true to the classic LSTM as possible while making changes to improve throughput. See the Languini Kitchen paper for details.

Run the following command to see if you can load the model and feed it a batch of dummy data.
```
CUDA_VISIBLE_DEVICES=0 python3 languini/projects/lstm/model.py
```

## Train the best config
The following configs achieve the best results for the 6h, 12h, 24h, 48h, and 96h compute scales. You can compute the number of steps given the compute class and the batch sizes with the load_throughput.ipynb notebook.
 

6h: qLSTM mini, bsz 80, seqlen 512 -> 94781 tokens per second, 49982 steps, 1 gradient accumulation step
```
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=localhost --master_port=12300 languini/projects/lstm/main.py mini \
  --train_batch_size 80 \
  --decay_steps 49982 \
  --max_train_steps 49982 \
  --gradient_accumulation_steps 1 \
  --tokens_per_second 94781 \
  --log_terminal_every 100 \
  --eval_every 1000 \
  --log_grads_every 1000 \
  --log_ckpt_every 1000
```

12h: qLSTM tiny, bsz 80, seqlen 512 -> 36930 tokens per second, 38950 steps, 2 gradient accumulation step
```
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=localhost --master_port=12300 languini/projects/lstm/main.py tiny \
  --train_batch_size 80 \
  --decay_steps 38950 \
  --max_train_steps 38950 \
  --gradient_accumulation_steps 2 \
  --tokens_per_second 36930 \
  --log_terminal_every 100 \
  --eval_every 500 \
  --log_grads_every 1000 \
  --log_ckpt_every 1000
```

24h: qLSTM small, bsz 84, seqlen 512 -> 11143 tokens per second, 22385 steps, 6 gradient accumulation step
```
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=localhost --master_port=12300 languini/projects/lstm/main.py small \
  --train_batch_size 84 \
  --decay_steps 22385 \
  --max_train_steps 22385 \
  --gradient_accumulation_steps 6 \
  --tokens_per_second 11143 \
  --log_terminal_every 100 \
  --eval_every 500 \
  --log_grads_every 1000 \
  --log_ckpt_every 1000
```

48h: qLSTM small, bsz 80, seqlen 512 -> 11143 tokens per second, 47010 steps, 5 gradient accumulation step
```
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=localhost --master_port=12300 languini/projects/lstm/main.py small \
  --train_batch_size 80 \
  --decay_steps 47010 \
  --max_train_steps 47010 \
  --gradient_accumulation_steps 5 \
  --tokens_per_second 11143 \
  --log_terminal_every 100 \
  --eval_every 500 \
  --log_grads_every 1000 \
  --log_ckpt_every 1000
```

96h: qLSTM small, bsz 160, seqlen 512 -> 11143 tokens per second, 47010 steps, 10 gradient accumulation step
```
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr=localhost --master_port=12300 languini/projects/lstm/main.py small \
  --train_batch_size 160 \
  --decay_steps 47010 \
  --max_train_steps 47010 \
  --gradient_accumulation_steps 10 \
  --tokens_per_second 11143 \
  --log_terminal_every 100 \
  --eval_every 500 \
  --log_grads_every 1000 \
  --log_ckpt_every 1000
```

## Evaluate
```
P=languini/projects/lstm/logs/quasiLSTM_bl16_books16384_bsz80_micro1_sl512_coslr0.0006to6e-06_h512_ff2048_nH8_dH32_nl4_clip0.0_decay49k_gpus1_defaultCompile_fp16_seed0
CUDA_VISIBLE_DEVICES=0 torchrun --standalone languini/projects/lstm/eval.py \
        --checkpoint_file "$P"/checkpoints/model.pt \
        --config_file "$P"/config.pickle \
        --eval_data_split test
```