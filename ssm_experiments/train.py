import os
import time
import sys
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW


def cpu_train(args, model, train_dataset, TO_TOKEN, tokenizer, device='cpu', checkpoint_dir=None, model_name=None, fixed_eval_dataset=None):
    print(f"Training on device: {device}")
    print("Masking answer region in inputs during training to prevent leakage.")

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    from transformers import get_scheduler
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=args.steps,
    )

    num_training_steps = args.steps
    model.train()

    losses = []
    training_examples = []
    accuracies = []
    accuracy_training_examples = []

    print("Pre-generating training data for consistent learning...")
    training_batches = []
    for i in range(min(num_training_steps, len(train_dataset))):
        batch = train_dataset[i]
        training_batches.append(batch)

    # Cleaner progress bar (no ASCII #### bar in logs)
    bar_format = (
        "{desc}: {n_fmt}/{total_fmt} | {rate_fmt} | "
        "{postfix} | Elapsed={elapsed} | ETA={remaining}"
    )
    progress_bar = tqdm(
        range(num_training_steps),
        desc='Training',
        dynamic_ncols=True,
        bar_format=bar_format,
        mininterval=0.5,
    )
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

        if args.model == "lstm":
            state = model.init_hidden(x_masked.size(0), device)
            logits, _ = model(x_masked, state)
        elif args.model == "mamba":
            logits = model(x_masked)[0]
        else:
            logits = model(x_masked)['logits']

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

        if checkpoint_dir and model_name and fixed_eval_dataset and (step + 1) % 50 == 0:
            print(f"\nSaving checkpoint at step {step + 1}...")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_step_{step + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)

            current_training_state = model.training
            model.eval()
            try:
                from .evaluate import offline_accuracy_evaluation
                with torch.no_grad():
                    accuracy = offline_accuracy_evaluation(model, fixed_eval_dataset, args, tokenizer, TO_TOKEN, device)
                    accuracies.append(accuracy)
                    accuracy_training_examples.append(step * args.train_batch_size)
                    print(f"Step {step + 1}: Accuracy = {accuracy:.1f}%")
            except Exception as e:
                print(f"Warning: Evaluation failed at step {step + 1}: {e}")
                accuracies.append(0.0)
                accuracy_training_examples.append(step * args.train_batch_size)
            finally:
                model.train(current_training_state)

        progress_bar.set_postfix_str(
            f"Loss={loss.item():.4f} | Examples={step * args.train_batch_size}", refresh=False
        )

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")

    return {
        'losses': losses,
        'training_examples': training_examples,
        'training_time': elapsed_time,
        'accuracies': accuracies,
        'accuracy_training_examples': accuracy_training_examples,
    }
