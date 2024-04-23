# Copyright 2022 The Languini Authors.
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
import math
import itertools
import torch
import sentencepiece as spm
import torch.nn.functional as F
import torch.distributed as dist

from tqdm import tqdm
from languini.train_lib import train_utils
from languini.common_lib import debug_utils
from languini.common_lib import common_utils
from languini.common_lib import parallel_utils
from languini.common_lib.debug_utils import check
from languini.train_lib.logger import CustomXAxisScalar


DEFAULT_CONFIG = {
    "device": None,
    "max_train_steps": None,
    "seq_len": None,
    "max_eval_steps": 50,
    "grad_clip_norm": 0.0,
    "gradient_accumulation_steps": 1,
    "tokens_per_second": 0,

    # logs
    "log_path": None,
    "log_terminal_every": None,     # print avg train loss to terminal 
    "log_metrics_every": None,      # log many train and time metrics      
    "eval_every": 1_000,            # run evaluation and log results
    "log_ckpt_every": 100,          # save model checkpoint
    "log_grads_every": 5_000,       # log gradients, weights, and step sizes to disk
    "log_activations_every": 5_000, # log model activations to disk
}


def evaluation(config, model, state, data_source, max_steps, last_n=-1, print_progress=False):
    """
    Evaluates the model on a datasource without gradient updates or extra logs besides loss.
    
    Args:
        config (Munch): an experiment config.
        model: the PyTorch model.
        state: the latest state of the model if it has one (or None to initialise a new one).
        data_source: the source for the input and target batches.
        max_step (int): number of batches do process for evaluation.
        last_n (int): evaluate loss on the last_n targets. If last_n is -1 it will evaluate on all targets.
        print_progress (bool): simple terminal log for eval.py to display progress.
    """
    c = config
    local_bsz = config.eval_batch_size // c.n_workers

    assert last_n <= c.seq_len and last_n != 0, "we cannot eval on the last_n=0 tokens or more tokens than there are in a sequence!"
    assert all(common_utils.flatten(common_utils.traverse(state, func=lambda x: x is None or x.shape[0] == 1))), "all state elements must have batch size 1!"
    assert max_steps == -1 or max_steps > 0, "Maximum number of steps has to be either -1 or a positive value."

    model.eval()
    data_source.reset()
    eval_batches = data_source if max_steps == -1 else itertools.islice(data_source, max_steps)

    batch_count = 0
    total_loss = 0
    total_top_k_counts = {}
    total_token_count = 0

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=c.device.type == "cuda"):
            # distribute the given state over the batch-size
            if state is None:
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    state = model.module.get_init_state(local_bsz, device=c.device)
                else:
                    state = model.get_init_state(local_bsz, device=c.device)
            else:
                # we distribute the same state across all batch elements
                state = common_utils.traverse(state, func=lambda x: torch.concatenate([x] * local_bsz) if x is not None else None)
            
            if print_progress:
                eval_batches = tqdm(eval_batches, total=None if max_steps == -1 else max_steps)

            # iterate over batches
            for batch_count, (batch_x, batch_y, is_padded) in enumerate(eval_batches, start=1):
                if print_progress and batch_count % 1_000 == 0:
                    parallel_utils.mprint(f"{batch_count:d}")
                
                assert batch_x.shape[0] == batch_y.shape[0] == 1, "evaluation requires microsteps-dim in batch to be 1"
                check(batch_x, (1, local_bsz, -1))

                # Remove the microbatch dimension
                batch_x, batch_y = batch_x.squeeze(0), batch_y.squeeze(0)
                bsz, seqlen = batch_x.shape
                
                # run the forward pass
                logits, state = model(batch_x, state, log=None)
                check(logits, (bsz, seqlen, c.vocab_size))

                # If last_n is positive we only care about the last_n losses and targets.
                if last_n == -1:
                    last_n = seqlen
                batch_x = batch_x[:, -last_n:]
                check(batch_x, (bsz, last_n))
                logits = logits[:, -last_n:, :]
                check(logits, (bsz, last_n, c.vocab_size))
                batch_y = batch_y[:, -last_n:]
                check(batch_y, (bsz, last_n))
                
                # compute loss
                logits = logits.reshape(bsz * last_n, c.vocab_size)
                batch_y = batch_y.reshape(bsz * last_n)
                all_losses = F.cross_entropy(input=logits, target=batch_y, reduction='none')
                check(all_losses, (bsz * last_n,))

                # mask losses that are padded (unlike training, evaluation can result in batches with padded batches)
                is_padding = batch_y.reshape(-1) == 0 # (bsz * last_n,)
                all_losses = all_losses.masked_fill(is_padding, 0.0)
                token_count = torch.sum(~is_padding)

                # compute accuracy for top-1 and top-10
                topk_counts = train_utils.total_correct(logits, batch_y, is_padding=is_padding, topk=(1, 10))
                for key in topk_counts.keys():
                    dist.all_reduce(topk_counts[key], dist.ReduceOp.SUM)
                    if key in total_top_k_counts.keys():
                        total_top_k_counts[key] += topk_counts[key].detach().item()
                    else:
                        total_top_k_counts[key] = topk_counts[key].detach().item()

                total_loss += torch.sum(all_losses).reshape((-1,)).detach()
                total_token_count += token_count.reshape((-1,))
        
        parallel_utils.mprint(f'total number of batches processed: {batch_count}')
        dist.all_reduce(total_loss, dist.ReduceOp.SUM)
        dist.all_reduce(total_token_count, dist.ReduceOp.SUM)

    return total_loss.item(), total_top_k_counts, total_token_count.item(), state


def log_eval_stats(eval_data_source, eval_steps, last_n, sp, logger, device):
    """Counts the number of eval batches and the length in string bytes. Saves these values for later."""
    eval_data_source.reset()
    eval_batches = eval_data_source if eval_steps == -1 else itertools.islice(eval_data_source, eval_steps)
    batch_count = 0

    micro_batches = eval_data_source.micro_batches
    local_micro_bsz = eval_data_source.bsz // eval_data_source.micro_batches
    seqlen = eval_data_source.seq_len

    # use tensors to count in case it is distributed across accelerators
    token_count = torch.zeros(1, device=device)
    str_length = torch.zeros(1, device=device)
    for batch_count, (batch_x, batch_y, is_padded) in enumerate(eval_batches, start=1):
        check(batch_x, (micro_batches, local_micro_bsz, seqlen))
        check(batch_y, (micro_batches, local_micro_bsz, seqlen))

        batch_x = batch_x[:, :, -last_n:]
        batch_y = batch_y[:, :, -last_n:]

        # decode targets and measure length
        batch_y = torch.reshape(batch_y, (batch_y.shape[0] * batch_y.shape[1], -1))
        str_lst = sp.decode(batch_y.cpu().tolist())
        for str in str_lst:
            str_length += len(str)

        # count non-padding tokens
        token_count += torch.sum(batch_y != 0) if is_padded else batch_y.numel()
    
    # sum across accelerators
    dist.all_reduce(str_length, dist.ReduceOp.SUM)
    dist.all_reduce(token_count, dist.ReduceOp.SUM)

    # convert to ints
    str_length = int(str_length.cpu().item())
    token_count = int(token_count.cpu().item())
    
    if parallel_utils.is_main_process() and logger:
        logger.log(
            {
                "eval_batches": batch_count,
                "eval_tokens": token_count,
                "eval_bytes": str_length,
            },
            step=None,
        )

        print(f"Eval batches: {batch_count:,}")
        print(f"Eval bytes: {str_length:,}")
    return str_length, batch_count, token_count


class LMTrainer:
    """A language modelling trainer. """
    
    def __init__(self, config, logger, model, opt, train_batches, eval_batches, scheduler=None):
        train_utils.check_config(config, DEFAULT_CONFIG)
        self.c = c = config
        self.logger = logger
        self.model = model.to(config.device)
        self.opt = opt
        self.scheduler = scheduler
        self.train_batches = train_batches
        self.eval_batches = eval_batches
        self.scaler = torch.cuda.amp.GradScaler(enabled=c.device.type == "cuda")

        # log hyperparameters
        train_utils.log_hyperparams(config, self.logger)

        # log total number of weights
        train_utils.print_model_size(self.model, self.c.vocab_size, self.c.h_dim, self.logger)

        # load model weights and state if a checkpoint is provided
        if "checkpoint_path" in self.c.keys() and self.c.checkpoint_path != "":
            self.model, self.curr_state = train_utils.load_checkpoint(model=self.model, path=c.checkpoint_path)
            print(f"Model checkpoint and state loaded from {c.checkpoint_path}")

        # load tokeniser
        self.sp = train_utils.load_tokeniser(config=c)
        assert self.c.vocab_size == self.sp.vocab_size(), f"config vocab size {c.vocab_size} doesn't match tokeniser vocab size {self.sp.vocab_size()}"

        # get number of eval steps and total eval bytes since that will be the same for every evaluation run
        parallel_utils.mprint("Measure evaluation data size ...")
        self.eval_bytes, _, _ = log_eval_stats(eval_data_source=self.eval_batches,
                                               eval_steps=self.c.max_eval_steps,
                                               sp=self.sp,
                                               logger=self.logger,
                                               device=self.c.device,
                                               last_n=self.c.seq_len)

    def train(self):
        c = self.c
        
        # StopWatches to track time spent doing different things
        load_watch = train_utils.StopWatch()        # batch loading
        forward_watch = train_utils.StopWatch()     # forward pass
        backward_watch = train_utils.StopWatch()    # backward pass
        train_watch = train_utils.StopWatch()       # train step
        eval_watch = train_utils.StopWatch()        # evaluation
        total_watch = train_utils.StopWatch()       # total step
        tokens_seen = 0                             # total number of tokens seen
        
        # we keep a separate model state for each micro step since gradient_accumulation_steps split the batch size
        curr_states = [None] * c.gradient_accumulation_steps
        
        total_watch.start()
        for step in range(c.max_train_steps):
            self.model.train()

            # boolean which tracks if during the current step we do some extra logging
            do_grads_log = c.log_grads_every > 0 and step % c.log_grads_every == 0 and step > 0
            do_activations_log = c.log_activations_every > 0 and step % c.log_activations_every == 0 and step > 0

            if not do_grads_log and not do_activations_log:
                # we only track time when we do no extra logging
                train_watch.start()
            
            # load the next training batch
            avg_loss = torch.tensor(0.0, device=c.device)

            load_watch.start()
            total_batch_x, total_batch_y, _ = next(self.train_batches)
            check(total_batch_x, (c.gradient_accumulation_steps, c.train_batch_size // c.gradient_accumulation_steps // c.n_workers, c.seq_len))
            load_watch.pause().count()

            for micro_step in range(c.gradient_accumulation_steps):
                curr_state = curr_states[micro_step]

                # trick taken from Karpathy's nanoGPT to not sync on every backward call
                self.model.require_backward_grad_sync = (micro_step == c.gradient_accumulation_steps - 1)
                
                # select the current micro batch
                batch_x = total_batch_x[micro_step]
                batch_y = total_batch_y[micro_step]
                check(batch_x, (c.train_batch_size // c.gradient_accumulation_steps // c.n_workers, c.seq_len))
                bsz, seqlen = batch_x.shape
                
                # run forward pass
                with torch.cuda.amp.autocast(enabled=c.device.type == "cuda"):
                    # get initial state
                    if curr_state is None:
                        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                            curr_state = self.model.module.get_init_state(bsz, device=c.device)
                        else:
                            curr_state = self.model.get_init_state(bsz, device=c.device)

                    # track state size
                    state_size = common_utils.get_total_tensor_size(curr_state)
                    
                    # perform the model forward pass with or without activation logs
                    if do_activations_log:
                        logits, curr_state = self.model(batch_x, curr_state, log=(self.logger, step))
                    else:
                        forward_watch.start()
                        logits, curr_state = self.model(batch_x, curr_state, log=None)
                        forward_watch.pause().count()                    
                    check(logits, (bsz, c.seq_len, c.vocab_size))

                    # compute loss
                    logits = logits.reshape(bsz * c.seq_len, c.vocab_size)
                    if do_grads_log:
                        debug_utils.log_stats_and_dist(logits, "Logits", log=(self.logger, step))
                    batch_y = batch_y.reshape(bsz * c.seq_len)
                    micro_avg_loss = F.cross_entropy(input=logits, target=batch_y).reshape((-1,))
                    check(micro_avg_loss, (1,))

                    # keep a sum of the avg_loss of each micro batch
                    avg_loss = avg_loss + micro_avg_loss.detach()
                    
                    # detach the current state
                    curr_state = common_utils.traverse(curr_state, func=lambda x: None if x is None else x.detach())

                    # log state stats
                    if do_grads_log:
                        curr_state_lst = common_utils.flatten(common_utils.traverse(curr_state, func=lambda x: None if x is None else x.cpu()))
                        for state_idx, state in enumerate(curr_state_lst):
                            if not state is None:
                                debug_utils.log_stats_and_dist(state, f"state{state_idx}", log=(self.logger, step))
                    
                    # check state size is constant
                    new_state_size = common_utils.get_total_tensor_size(curr_state)
                    if state_size != 0:
                        assert state_size == new_state_size, f"After forward call state size changed from {state_size} to {new_state_size}"
                    
                # write the state of this microstep into the list of states
                curr_states[micro_step] = curr_state

                # scale loss in case of lower precision and perform the backward pass
                backward_watch.start()

                # we need to divide the loss by the number of gradient acc. steps to get the average gradient before we step below
                micro_avg_loss = micro_avg_loss / c.gradient_accumulation_steps
                if self.scaler:
                    self.scaler.scale(micro_avg_loss).backward()
                else:
                    micro_avg_loss.backward()
                backward_watch.pause().count()

            # collect the avg_loss over all micro steps and devices and compute the average
            dist.reduce(avg_loss, op=dist.ReduceOp.SUM, dst=0)
            avg_loss = avg_loss.detach().item() / (parallel_utils.WORLD_SIZE * c.gradient_accumulation_steps)
            
            # unscale gradients before clipping and logging
            if self.scaler:
                self.scaler.unscale_(self.opt)

            # clip gradients if clip is larger than 0.0
            if c.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip_norm)

            # perform an optimiser step and log weights, gradients, and step sizes
            if do_grads_log:
                debug_utils.log_weight_stats(self.model, "Weights", log=(self.logger, step))
                debug_utils.log_gradient_stats(self.model, "Grads", log=(self.logger, step))
                debug_utils.step_and_log_diff(self.scaler, self.opt, self.model, "Step", log=(self.logger, step))
            elif self.scaler:
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            
            # learning rate schedule has to step too
            if self.scheduler:
                self.scheduler.step()

            tokens_seen += c.train_batch_size * c.seq_len
            
            if not do_grads_log and not do_activations_log:
                train_watch.pause().count()

            # log train loss to terminal
            if c.log_terminal_every > 0 and step % c.log_terminal_every == 0 and parallel_utils.is_main_process():
                print(f"step={step:6d}  loss={avg_loss:0.5f}", flush=True)

            # Perfom a validation set evaluation (fixed number of batches)
            if c.eval_every > 0 and step % c.eval_every == 0 and step > 0:
                eval_watch.start()
                self.validation(curr_state=curr_states[0], step=step)
                eval_watch.pause().count()

            # Write logs to disk
            if parallel_utils.is_main_process() and c.log_metrics_every > 0 and step % c.log_metrics_every == 0 and step > 0:
                
                # log general speed metrics - only do so when we are not doing some extra logging (activations or gradients)
                if not do_grads_log and not do_activations_log:
                    tokens_per_batch = total_batch_x.numel()
                    load_time_per_batch = load_watch.read()
                    forward_time_per_batch = forward_watch.read()
                    backward_time_per_batch = backward_watch.read()
                    step_time_per_batch = train_watch.read()
                    total_time_per_batch = total_watch.read()
                    eval_time = eval_watch.read()
                    tokens_per_second = tokens_per_batch / step_time_per_batch
                    iter_per_second = 1. / step_time_per_batch
                    
                    print(f"tokens per second: {round(tokens_per_second):,}")

                    self.logger.log(
                        {
                            "_time/tokens_per_second": tokens_per_second,
                            "_time/iterations_per_second": iter_per_second,
                            "_time/load_batch": load_time_per_batch,
                            "_time/forward": forward_time_per_batch,
                            "_time/backward": backward_time_per_batch,
                            "_time/train_step": step_time_per_batch,
                            "_time/total_step": total_time_per_batch,
                        },
                        step
                    )

                    if eval_time > 0:
                        self.logger.log({"_time/eval": eval_time}, step)

                self.logger.log(
                    {
                        "_train/loss": avg_loss,
                        "_train/tokens_seen": tokens_seen,
                    },
                    step
                )
                curr_lrs = [pg['lr'] for pg in self.opt.param_groups]
                for idx, lr in enumerate(curr_lrs):  # lr for each param group
                    self.logger.log({f"_train/learning_rate_{idx}": lr}, step)

            # Write the current model weights to disk
            if parallel_utils.is_main_process() and c.log_ckpt_every > 0 and step % c.log_ckpt_every == 0 and step > 0:
                self.save_checkpoint(self.logger, step)

            total_watch.count()
        
        # Final validation run and checkpoint
        self.validation(curr_state=curr_states[0], step=step)
        if parallel_utils.is_main_process():
            self.save_checkpoint(self.logger, step)

    def validation(self, curr_state, step):
        """Run the model on the test data."""
        c = self.c
        eval_state = common_utils.traverse(curr_state, lambda x: x[:1] if x is not None else None)
        eval_total_loss, eval_total_topk, eval_token_count, _ = evaluation(config=c,
                                                                           model=self.model,
                                                                           state=eval_state,
                                                                           data_source=self.eval_batches,
                                                                           max_steps=c.max_eval_steps)
        # loss and ppl over number of tokens
        eval_avg_loss = eval_total_loss / eval_token_count
        eval_ppl = math.exp(eval_avg_loss)
        # loss and ppl over number of bytes
        eval_norm_loss = eval_total_loss / self.eval_bytes
        eval_norm_ppl = math.exp(eval_norm_loss)
        # accuracy over tokens
        eval_topk_accs = {key: eval_total_topk[key] / eval_token_count for key in eval_total_topk.keys()}

        if parallel_utils.is_main_process():
            number_of_tokens = (step + 1) * c.train_batch_size * c.seq_len # +1 as steps 0-indexed
            theoretical_gpu_seconds = number_of_tokens / c.tokens_per_second if c.tokens_per_second > 0 else 0  
            # Note, you cannot log floating point 'steps' so you cannot compute gpu hours here.
            def log_over_all_axes(name, value):
                """Logs value over steps, tokens, and gpu seconds."""
                metrics = {
                    name: value,
                    f"{name}_over_tokens": CustomXAxisScalar(value, axis_name="n_tokens", axis_val=number_of_tokens),
                    f"{name}_over_gpuseconds": CustomXAxisScalar(value, axis_name="gpu_seconds", axis_val=theoretical_gpu_seconds),
                }
                self.logger.log(metrics, step)

            log_over_all_axes("_eval/normalised_loss", eval_norm_loss)
            self.logger.log(
                {
                    "_eval/loss": eval_avg_loss,
                    "_eval/total_loss": eval_total_loss,
                },
                step
            )

            # skip ppl logging for initial loss which skews the plot unnecessarily
            if eval_ppl < 1_000:
                log_over_all_axes("_eval/ppl", eval_ppl)    
            if eval_norm_ppl < 1_000:
                log_over_all_axes("_eval/normalised_ppl", eval_norm_ppl)
            for key in eval_topk_accs:
                log_over_all_axes(f"_eval/top{key}_acc", eval_topk_accs[key])
            print(f"EVAL step={step:d} loss={eval_avg_loss:0.5f} acc={eval_topk_accs[1]:0.5f}") 

    def save_checkpoint(self, logger, step):
        """Saves a checkpoint of the current model to disk. """

        def _save_checkpoint(path):
            # create folder if it doesn't exist
            if not os.path.exists(path):
                os.makedirs(path)

            # remove previous log file there is one        
            file = os.path.join(path, "model.pt")
            if os.path.exists(file):
                os.remove(file)

            # Write checkpoint
            with open(file, 'wb') as f:
                torch.save({
                    "step": step,
                    "model_state_dict": self.model.state_dict(),
                    "opt_state_dict": self.opt.state_dict(),
                    }, f)
            print(f"Checkpoint written at step {step} to:\n{file}")

        if logger.use_tb:
            ckpt_path = os.path.join(logger.log_path, "checkpoints")
            _save_checkpoint(ckpt_path)

        if logger.use_wandb:
            ckpt_path = os.path.join(logger.wandb_run_dir, "checkpoints")
            _save_checkpoint(ckpt_path)
