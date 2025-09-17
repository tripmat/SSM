import os
import json
import argparse
import sys
import io
from datetime import datetime
import atexit
import random
import numpy as np
import torch

from .models.registry import get_model
from .data import get_tokenizer, get_train_dataset, CopyDataset
from .train import cpu_train
from .evaluate import offline_accuracy_evaluation, evaluate_length_generalization, generate_fixed_eval_dataset
from .utils.parameter_matching import get_optimal_configs, count_parameters
from .utils.plotting import plot_paper_reproduction, create_comparison_table


def main():
    parser = argparse.ArgumentParser(description="Transformer vs Mamba (slim) on synthetic copy task")
    parser.add_argument('--steps', type=int, default=2000, help='Training steps')
    parser.add_argument('--lr', type=float, default=1.25e-4, help='Learning rate')
    parser.add_argument('--train-batch-size', type=int, default=4, help='Train batch size')
    parser.add_argument('--eval-batch-size', type=int, default=4, help='Eval batch size')
    parser.add_argument('--eval-num-batches', type=int, default=3, help='Eval batches per length')
    parser.add_argument('--min-train-len', type=int, default=5)
    parser.add_argument('--max-train-len', type=int, default=50)
    parser.add_argument('--min-eval-len', type=int, default=10)
    parser.add_argument('--max-eval-len', type=int, default=200)
    parser.add_argument('--context-len', type=int, default=220)
    parser.add_argument('--eval-context-len', type=int, default=450)
    parser.add_argument('--outputs', type=str, default='outputs', help='Outputs root directory')
    parser.add_argument('--only', type=str, choices=['both', 'transformer', 'mamba'], default='both',
                        help='Select which model(s) to run')
    args_cli = parser.parse_args()

    # Reproducibility: seed all relevant RNGs; allow override via env var
    seed = int(os.environ.get('SSM_SEED', '1337'))
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        print(f"Using seed: {seed}")
    except Exception as e:
        print(f"Warning: failed to set reproducible seeds: {e}")

    # Set up run log: tee stdout/stderr to outputs/logs/run_YYYYmmdd_HHMMSS.log
    outputs_dir = os.path.abspath(args_cli.outputs)
    if os.path.islink(outputs_dir):
        raise ValueError(f"Refusing to write to symlinked outputs directory: {outputs_dir}")
    os.makedirs(outputs_dir, exist_ok=True)
    logs_dir = os.path.join(outputs_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(logs_dir, f'run_{ts}.log')

    class Tee(io.TextIOBase):
        def __init__(self, *streams):
            self.streams = streams
        def write(self, s):
            for st in self.streams:
                st.write(s)
                st.flush()
            return len(s)
        def flush(self):
            for st in self.streams:
                st.flush()

    log_fh = open(log_path, 'w', buffering=1)
    sys.stdout = Tee(sys.stdout, log_fh)
    sys.stderr = Tee(sys.stderr, log_fh)

    @atexit.register
    def _close_log():
        try:
            log_fh.flush()
            log_fh.close()
        except Exception:
            pass

    class Args:
        def __init__(self):
            self.train_task = "copy"
            self.eval_task = "copy"
            self.vocab_size = 26
            self.n_gram = 0
            self.length_answer = 0

            self.hidden_size = 256
            self.layers = 6
            self.heads = 8
            self.num_masked_heads = 4
            self.state_dim = 16

            self.lr = args_cli.lr
            self.epochs = 1
            self.steps = args_cli.steps
            self.train_batch_size = args_cli.train_batch_size
            self.eval_batch_size = args_cli.eval_batch_size
            self.eval_num_batches = args_cli.eval_num_batches

            self.min_train_len = args_cli.min_train_len
            self.max_train_len = args_cli.max_train_len
            self.min_eval_len = args_cli.min_eval_len
            self.max_eval_len = args_cli.max_eval_len
            self.context_len = args_cli.context_len
            self.eval_context_len = args_cli.eval_context_len

    results = {}
    configs = get_optimal_configs()
    # Optional filtering to run a single model
    if args_cli.only == 'transformer':
        configs = {k: v for k, v in configs.items() if k == 'transformer_rope'}
    elif args_cli.only == 'mamba':
        configs = {k: v for k, v in configs.items() if k == 'mamba'}

    args_temp = Args()
    tokenizer, TO_TOKEN, TO_CHAR = get_tokenizer(args_temp)
    fixed_eval_dataset = generate_fixed_eval_dataset(args_temp, tokenizer, TO_TOKEN, num_examples=100)

    print("Starting experiments...")
    print("=" * 60)

    for model_name, config in configs.items():
        print(f"\nTraining {model_name.upper()}...")
        print("-" * 40)

        args = Args()
        args.model = config['model']
        args.hidden_size = config['hidden_size']
        args.layers = config['layers']
        if 'heads' in config:
            args.heads = config['heads']
        if 'state_dim' in config:
            args.state_dim = config['state_dim']

        model = get_model(args, tokenizer)
        param_count = count_parameters(model)
        print(f"Model parameters: {param_count:,}")

        # Outputs layout
        checkpoints_dir = os.path.join(outputs_dir, 'checkpoints')
        figures_dir = os.path.join(outputs_dir, 'figures')
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

        # Train dataset
        train_dataset = CopyDataset(
            tokenizer=tokenizer,
            vocab_size=args.vocab_size,
            n_gram=args.n_gram,
            length_answer=args.length_answer,
            train_task=args.train_task,
            sequence_length=args.context_len,
            min_length=args.min_train_len,
            max_length=args.max_train_len,
            num_examples=1000,
            batch_size=args.train_batch_size,
        )

        # Train
        training_results = cpu_train(
            args,
            model,
            train_dataset,
            TO_TOKEN,
            tokenizer,
            device='cpu',
            checkpoint_dir=checkpoints_dir,
            model_name=model_name,
            fixed_eval_dataset=fixed_eval_dataset,
        )

        # Save final model checkpoint BEFORE evaluation, then reload for eval to ensure parity
        config_str = f"{model_name}_{args.hidden_size}h_{args.layers}l_{args.lr}lr_{args.steps}s"
        checkpoint_path = os.path.join(checkpoints_dir, f"{config_str}.pth")
        results_path = os.path.join(checkpoints_dir, f"{config_str}_results.json")

        try:
            import torch
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved final checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Warning: failed to save model checkpoint: {e}")

        # Recreate model and load saved weights for evaluation (parity with post-run evals)
        try:
            eval_model = get_model(args, tokenizer)
            # Safe load: prefer weights_only when available; validate type
            try:
                state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)  # type: ignore[arg-type]
            except TypeError:
                state_dict = torch.load(checkpoint_path, map_location='cpu')
            if not isinstance(state_dict, dict) or not all(hasattr(v, 'size') for v in state_dict.values()):
                raise ValueError("Unexpected checkpoint content (expected a state_dict mapping)")
            eval_model.load_state_dict(state_dict)
            print("Reloaded model from final checkpoint for evaluation.")
        except Exception as e:
            print(f"Warning: failed to reload model from checkpoint: {e}. Using in-memory model for eval.")
            eval_model = model

        # Evaluate using reloaded model
        fixed_accuracy = offline_accuracy_evaluation(eval_model, fixed_eval_dataset, args, tokenizer, TO_TOKEN)
        length_gen_results = evaluate_length_generalization(args, eval_model, tokenizer, TO_TOKEN)

        results_obj = {
            'training': training_results,
            'fixed_accuracy': fixed_accuracy,
            'length_gen': length_gen_results,
            'param_count': param_count,
            'config': {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))},
        }
        with open(results_path, 'w') as f:
            serializable_results = {
                'training': {
                    'losses': [float(x) for x in results_obj['training']['losses']],
                    'training_examples': results_obj['training']['training_examples'],
                    'training_time': results_obj['training']['training_time'],
                    'accuracies': results_obj['training']['accuracies'],
                    'accuracy_training_examples': results_obj['training']['accuracy_training_examples'],
                },
                'fixed_accuracy': results_obj['fixed_accuracy'],
                'length_gen': results_obj['length_gen'],
                'param_count': int(results_obj['param_count']),
                'config': results_obj['config'],
            }
            json.dump(serializable_results, f, indent=2)

        results[model_name] = results_obj
        print(f"Completed {model_name}")
        print(f"Final training loss: {results_obj['training']['losses'][-1]:.4f}")
        print(f"Training time: {results_obj['training']['training_time']:.1f}s")

    # Plots and summary
    print(f"\nGenerating plots in {figures_dir}...")
    has_t = 'transformer_rope' in results
    has_m = 'mamba' in results
    if has_t and has_m:
        plot_paper_reproduction(results['transformer_rope'], results['mamba'],
                                os.path.join(figures_dir, 'paper_reproduction.png'))
        from .utils.plotting import plot_comprehensive_summary, plot_comparison
        plot_comprehensive_summary(results['transformer_rope'], results['mamba'],
                                   os.path.join(figures_dir, 'comprehensive_summary.png'))
        plot_comparison(results['transformer_rope'], results['mamba'],
                        os.path.join(figures_dir, 'comparison_overview.png'))
        create_comparison_table(results['transformer_rope'], results['mamba'])
    elif has_t:
        from .utils.plotting import plot_length_gen_single, plot_comprehensive_summary
        plot_length_gen_single(results['transformer_rope'], 'Transformer: RoPE',
                               os.path.join(figures_dir, 'transformer_length_generalization.png'))
        # Single-model summary (Mamba=None)
        plot_comprehensive_summary(results['transformer_rope'], None,
                                   os.path.join(figures_dir, 'transformer_comprehensive_summary.png'))
        print("Only Transformer run; comparison plot skipped.")
    elif has_m:
        from .utils.plotting import plot_length_gen_single, plot_comprehensive_summary
        plot_length_gen_single(results['mamba'], 'GSSM: Mamba',
                               os.path.join(figures_dir, 'mamba_length_generalization.png'))
        plot_comprehensive_summary(None, results['mamba'],
                                   os.path.join(figures_dir, 'mamba_comprehensive_summary.png'))
        print("Only Mamba run; comparison plot skipped.")

    results_file = os.path.join(outputs_dir, "experiment_results.json")
    with open(results_file, 'w') as f:
        serializable_results = {}
        for name, r in results.items():
            serializable_results[name] = {
                'training': {
                    'losses': [float(x) for x in r['training']['losses']],
                    'training_examples': r['training']['training_examples'],
                    'training_time': r['training']['training_time'],
                    'accuracies': r['training']['accuracies'],
                    'accuracy_training_examples': r['training']['accuracy_training_examples'],
                },
                'fixed_accuracy': r['fixed_accuracy'],
                'length_gen': r['length_gen'],
                'param_count': r['param_count'],
                'config': r['config'],
            }
        json.dump(serializable_results, f, indent=2)
    print(f"Detailed results saved to: {results_file}")


if __name__ == "__main__":
    main()
