import numpy as np
import torch
from types import SimpleNamespace

from .data.datasets import get_eval_dataset


def offline_accuracy_evaluation(model, fixed_eval_dataset, args, tokenizer, TO_TOKEN, device='cpu'):
    print(f"Running offline accuracy evaluation on {len(fixed_eval_dataset)} examples (masked answer region)...")
    model.eval()
    pad_id = TO_TOKEN['*']
    with torch.inference_mode():
        all_string_accuracies = []
        for eval_example in fixed_eval_dataset:
            # Original (unmasked) inputs used for computing ground-truth span
            eval_x = eval_example['input_ids'].to(device)
            eval_mask_full = eval_example['mask'].to(device)

            # Mask the answer region to avoid teacher-forcing leakage
            eval_x_masked = eval_x.clone()
            # Mask only positions that are themselves in the answer region (avoid masking '|')
            eval_mask_x = eval_mask_full[:, :-1]
            eval_x_masked[:, :-1][eval_mask_x == 1] = pad_id

            # Forward pass on masked inputs
            if args.model == "lstm":
                eval_state = model.init_hidden(eval_x_masked.size(0), device)
                eval_logits, _ = model(eval_x_masked, eval_state)
            elif args.model == "mamba":
                eval_logits = model(eval_x_masked)[0]
            else:
                try:
                    eval_logits = model(eval_x_masked, use_cache=False)['logits']
                except Exception:
                    eval_logits = model(eval_x_masked)['logits']

            # Predictions for the answer region
            eval_pred = torch.argmax(eval_logits, dim=-1)

            # Compute accuracy against the original unmasked input
            str_acc, _ = get_score(args, tokenizer, eval_x, eval_pred, 0)
            all_string_accuracies.append(str_acc)

        avg_accuracy = sum(all_string_accuracies) / len(all_string_accuracies) * 100
        print(f"Fixed dataset accuracy: {avg_accuracy:.1f}% ({len(all_string_accuracies)} examples)")
        return avg_accuracy


def evaluate_length_generalization(args, model, tokenizer, TO_TOKEN, device='cpu'):
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
                if args.model == "lstm":
                    state = model.init_hidden(args.eval_batch_size, device)
                    logits, state = model(x_masked, state)
                elif args.model == "mamba":
                    logits = model(x_masked)[0]
                else:
                    logits = model(x_masked)['logits']
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


def generate_fixed_eval_dataset(args, tokenizer, TO_TOKEN, num_examples=100):
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
