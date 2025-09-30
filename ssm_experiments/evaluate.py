import numpy as np
import math
import torch
from types import SimpleNamespace

from .data.datasets import get_eval_dataset


def offline_accuracy_evaluation(model, fixed_eval_dataset, args, tokenizer, TO_TOKEN, device=None, batch_size=None, log_interval: int = 0):
    """Evaluate exact-match accuracy on a fixed set, batched.

    - Masks the answer region in the inputs to avoid leakage.
    - Runs inference in batches for speed and stability.
    - Returns percentage accuracy (0–100).
    """
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Determine batch size
    if batch_size is None:
        batch_size = getattr(args, 'eval_batch_size', 4)

    num_examples = len(fixed_eval_dataset)
    if log_interval:
        print(f"Running offline accuracy evaluation on {num_examples} examples (batched, masked answer region)...")

    model.eval()
    model = model.to(device)  # Ensure model is on correct device
    pad_id = TO_TOKEN['*']

    # Stack all examples into tensors for efficient slicing
    with torch.inference_mode():
        all_inputs = torch.cat([ex['input_ids'] for ex in fixed_eval_dataset], dim=0)
        all_masks = torch.cat([ex['mask'] for ex in fixed_eval_dataset], dim=0)

        total_correct = 0
        # Iterate in batches
        total_batches = math.ceil(num_examples / batch_size)
        for bidx, start in enumerate(range(0, num_examples, batch_size), start=1):
            end = min(start + batch_size, num_examples)
            x_full = all_inputs[start:end].to(device)
            mask_full = all_masks[start:end].to(device)

            # Create masked input to avoid leakage (mask the answer tokens themselves)
            x_masked = x_full.clone()
            mask_x = mask_full[:, :-1]
            x_masked[:, :-1][mask_x == 1] = pad_id

            # Forward pass (handle dict/tuple outputs)
            try:
                out = model(x_masked, use_cache=False)
            except Exception:
                out = model(x_masked)
            if isinstance(out, dict):
                logits = out.get('logits', out)
            elif isinstance(out, (list, tuple)):
                logits = out[0]
            else:
                logits = out

            pred = torch.argmax(logits, dim=-1)

            # Compute per-sample exact match within the batch
            batch_correct = 0
            for i in range(len(x_full)):
                str_acc, _ = get_score(args, tokenizer, x_full, pred, i)
                batch_correct += int(str_acc)
            total_correct += batch_correct

            if log_interval and (bidx % log_interval == 0 or bidx == total_batches):
                processed = end
                running_acc = (total_correct / processed) * 100.0
                print(f"  eval batch {bidx}/{total_batches} | processed={processed}/{num_examples} | running_acc={running_acc:.2f}%")

        avg_accuracy = (total_correct / num_examples) * 100.0
        if log_interval:
            print(f"Fixed dataset accuracy: {avg_accuracy:.1f}% ({num_examples} examples)")
        return avg_accuracy


def evaluate_length_generalization(args, model, tokenizer, TO_TOKEN, device=None):
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model = model.to(device)
    lengths = np.arange(args.min_eval_len, args.max_eval_len)
    results = []
    print("Evaluating length generalization with masked answer region...\n")

    for ood_length in lengths:
        if hasattr(args, 'eval_context_len') and ood_length > args.eval_context_len:
            print(f"Skipping length {ood_length} (exceeds context length {args.eval_context_len})")
            continue
        str_acc_batch = np.zeros(args.eval_num_batches)
        char_acc_mean = 0
        for jj in range(args.eval_num_batches):
            # Build a non-mutating view of args for evaluation dataset construction
            eval_seq_len = getattr(args, 'eval_context_len', getattr(args, 'context_len', 220))
            eval_args = SimpleNamespace(
                vocab_size=args.vocab_size,
                n_gram=args.n_gram,
                length_answer=args.length_answer,
                eval_task=args.eval_task,
                eval_batch_size=args.eval_batch_size,
                context_len=eval_seq_len,
            )
            long_dataset = get_eval_dataset(
                eval_args, tokenizer, TO_TOKEN,
                target_min_len=ood_length,
                target_max_len=ood_length
            )
            batch = next(iter(long_dataset))
            # Keep original input for span indexing and ground-truth
            x_full = batch['input_ids'].to(device)
            mask_full = batch['mask'].to(device)

            # Create masked input to avoid leakage
            x = x_full
            x_masked = x_full.clone()
            mask_x = mask_full[:, :-1]
            x_masked[:, :-1][mask_x == 1] = TO_TOKEN['*']
            with torch.no_grad():
                out = model(x_masked)
                if isinstance(out, dict):
                    logits = out.get('logits', out)
                elif isinstance(out, (list, tuple)):
                    logits = out[0]
                else:
                    logits = out
                pred = torch.argmax(logits, dim=-1)
                for i in range(len(x)):
                    str_acc, char_acc = get_score(args, tokenizer, x, pred, i)
                    str_acc_batch[jj] += str_acc
                    char_acc_mean += char_acc
        str_acc_batch = str_acc_batch / len(x)
        mean_str_acc = np.mean(str_acc_batch)
        std_str_acc = np.std(str_acc_batch)
        mean_char_acc = char_acc_mean / (len(x) * args.eval_num_batches)
        print(f"{args.eval_task}; len {ood_length}: {mean_str_acc} +- {std_str_acc}; char: {mean_char_acc}")
        avg_accuracy = mean_str_acc * 100
        results.append({'length': int(ood_length), 'accuracy': float(avg_accuracy)})
    print("\n")
    return results


def generate_fixed_eval_dataset(args, tokenizer, TO_TOKEN, num_examples=1000, verbose: bool = True):
    if verbose:
        print(f"Generating fixed evaluation dataset with {num_examples} examples of length {args.max_train_len}...")
    eval_dataset = get_eval_dataset(
        args, tokenizer, TO_TOKEN,
        target_min_len=args.max_train_len,
        target_max_len=args.max_train_len
    )
    fixed_eval_examples = []
    batches_needed = (num_examples + args.eval_batch_size - 1) // args.eval_batch_size
    for _ in range(batches_needed):
        if len(fixed_eval_examples) >= num_examples:
            break
        batch = next(iter(eval_dataset))
        for i in range(len(batch['input_ids'])):
            if len(fixed_eval_examples) < num_examples:
                fixed_eval_examples.append({
                    'input_ids': batch['input_ids'][i:i+1],
                    'mask': batch['mask'][i:i+1],
                    'input': [batch['input'][i]],
                })
    if verbose:
        print(f"Generated {len(fixed_eval_examples)} evaluation examples")
    return fixed_eval_examples


def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def get_score(args, tokenizer, x, pred, i):
    x_out = tokenizer.decode(x[i])
    x_out = x_out.split('.')[0] + '.'
    pred_out = tokenizer.decode(pred[i])
    if args.eval_task == "prefix_ngram":
        index = find(x_out, '|')[-1]
    elif args.eval_task in ["suffix_ngram", "copy", "duplicate_ngram"]:
        index = x_out.index('|')
    if args.eval_task == "suffix_ngram":
        gt = x_out[index + 1 + args.n_gram:][:-1]
        start_idx = index + args.n_gram
    else:
        gt = x_out[index + 1:][:-1]
        start_idx = index
    end_idx = start_idx + len(gt)
    pred_model = pred_out[start_idx:end_idx]
    str_acc = int(gt == pred_model)
    char_acc = sum(map(str.__eq__, gt, pred_model)) / max(len(gt), len(pred_model))
    return str_acc, char_acc
