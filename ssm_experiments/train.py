import os
import time
import math
import sys
import random
import numpy as np
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW


def train_model(args, model, train_dataset, TO_TOKEN, tokenizer, device=None, checkpoint_dir=None, model_name=None, fixed_eval_dataset=None):
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Training on device: {device}")
    print("Masking answer region in inputs during training to prevent leakage.")

    model = model.to(device)
    # Diagnostics: confirm device placement
    try:
        print(f"Model device: {next(model.parameters()).device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except StopIteration:
        # Model with no parameters
        print("Model has no parameters to inspect for device.")
    # Optimizer uses peak LR; scheduler scales it per step
    max_lr = float(getattr(args, 'max_lr', getattr(args, 'lr', 1e-4)))
    min_lr = float(getattr(args, 'min_lr', 1e-6))
    warmup_steps = int(getattr(args, 'warmup_steps', 300))
    optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=0.1)

    # Cosine schedule with warmup: 0 -> max_lr (warmup) -> min_lr (cosine decay)
    total_steps = int(args.steps)
    min_ratio = min_lr / max_lr if max_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup from 0 to 1
            return float(step) / max(1, warmup_steps)
        if step >= total_steps:
            return min_ratio
        # Cosine decay from 1 to min_ratio
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
        return min_ratio + (1.0 - min_ratio) * cos_decay

    from torch.optim.lr_scheduler import LambdaLR
    lr_scheduler = LambdaLR(optimizer, lr_lambda)

    num_training_steps = args.steps
    model.train()

    losses = []
    training_examples = []
    accuracies = []
    accuracy_training_examples = []
    lr_schedule = []
    lr_training_examples = []

    print("Pre-generating training data for consistent learning...")
    training_batches = []
    for i in range(min(num_training_steps, len(train_dataset))):
        batch = train_dataset[i]
        training_batches.append(batch)

    # Cleaner progress bar (no ASCII #### bar in logs)
    bar_format = (
        "{desc} | Elapsed={elapsed} | ETA={remaining}"
    )
    # Friendly label for the training progress bar
    pretty_name = None
    if model_name:
        if model_name.startswith('transformer_'):
            suffix = model_name.split('transformer_', 1)[1]
            mapping = {
                'rope': 'Rope',
                'nope': 'NoPE',
                'alibi': 'ALiBi',
                'hard_alibi': 'Hard-ALiBi',
            }
            pretty_name = mapping.get(suffix, suffix.title())
        elif model_name == 'paper_mamba':
            pretty_name = 'Paper-Mamba'
        elif model_name == 'minimal_mamba':
            pretty_name = 'Minimal-Mamba'
        else:
            pretty_name = model_name.title()
    else:
        pretty_name = getattr(args, 'model', 'Model')

    # Route tqdm output to real TTY only (avoid spamming log file via Tee)
    tqdm_stream = getattr(sys, '__stderr__', None) or sys.stderr
    try:
        is_tty = bool(tqdm_stream.isatty())
    except Exception:
        is_tty = False

    progress_bar = tqdm(
        range(num_training_steps),
        desc=f'Training {pretty_name}',
        dynamic_ncols=True,
        bar_format=bar_format,
        mininterval=1.0,
        leave=False,
        file=tqdm_stream,
        disable=not is_tty,
    )

    # Initialize description to avoid comma formatting issues
    if is_tty:
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_description_str(f"Training {pretty_name}: 0/{num_training_steps} | 0.00it/s | LR={current_lr:.2e} | Loss=0.0000 | Examples=0", refresh=False)
    start_time = time.time()

    pad_id = TO_TOKEN['*']

    for step in progress_bar:
        batch = training_batches[step % len(training_batches)]

        x_full = batch['input_ids'].to(device)
        m_full = batch['mask'].to(device)

        # Causal LM shift
        x = x_full[:, :-1]
        y = x_full[:, 1:]

        # Split masks for inputs vs targets to avoid masking the '|' boundary
        mask_x = m_full[:, :-1]   # positions of answer tokens in inputs
        mask_y = m_full[:, 1:]    # positions of answer tokens in targets

        # Mask the answer region in the inputs to avoid teacher-forcing leakage
        # We mask tokens that are themselves in the answer region (mask_x).
        # This keeps the '|' boundary visible while hiding answer tokens.
        x_masked = x.clone()
        x_masked[mask_x == 1] = pad_id

        optimizer.zero_grad()

        out = model(x_masked)
        if isinstance(out, dict):
            logits = out.get('logits', out)
        elif isinstance(out, (list, tuple)):
            logits = out[0]
        else:
            logits = out

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"Warning: NaN/Inf in logits at step {step}")
            print(f"  logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}")
            continue

        if isinstance(logits, dict):
            logits = logits['logits']

        shift_labels = y.contiguous()
        shift_logits = logits.contiguous()
        mask_flat = mask_y.contiguous().view(-1)

        loss_fct = CrossEntropyLoss(ignore_index=TO_TOKEN['*'], reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        mask_sum = torch.sum(mask_flat)
        if mask_sum == 0:
            print(f"Warning: mask_sum is 0 at step {step}")
            loss = torch.tensor(0.0, device=shift_logits.device, requires_grad=True)
        else:
            loss = torch.sum(loss * mask_flat) / mask_sum

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: loss is {loss.item()} at step {step}")
            loss = torch.tensor(3.0, device=shift_logits.device, requires_grad=True)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"Warning: grad_norm is {grad_norm} at step {step}")
            optimizer.zero_grad()
        else:
            optimizer.step()
        lr_scheduler.step()

        losses.append(loss.item())
        training_examples.append(step * args.train_batch_size)

        # Track learning rate schedule
        current_lr = optimizer.param_groups[0]['lr']
        lr_schedule.append(current_lr)
        lr_training_examples.append(step * args.train_batch_size)

        if checkpoint_dir and model_name and (step + 1) % 100 == 0:
            print(f"\nSaving checkpoint at step {step + 1}...")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_step_{step + 1}.pth")

            # Save comprehensive checkpoint including all model state
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'step': step + 1,
                'loss': loss.item(),
                'torch_rng_state': torch.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'random_rng_state': random.getstate(),
            }

            # Add CUDA RNG state if using GPU
            if device == 'cuda':
                checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state()
                checkpoint['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()

            torch.save(checkpoint, checkpoint_path)

        if is_tty:
            current_lr = optimizer.param_groups[0]['lr']
            rate = progress_bar.format_dict.get('rate', 0) or 0
            progress_bar.set_description_str(
                f"Training {pretty_name}: {step+1}/{num_training_steps} | {rate:.2f}it/s | LR={current_lr:.2e} | Loss={loss.item():.4f} | Examples={step * args.train_batch_size}", refresh=False
            )

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")

    return {
        'losses': losses,
        'training_examples': training_examples,
        'training_time': elapsed_time,
        'accuracies': accuracies,
        'accuracy_training_examples': accuracy_training_examples,
        'lr_schedule': lr_schedule,
        'lr_training_examples': lr_training_examples,
    }


# Backward compatibility alias
cpu_train = train_model
