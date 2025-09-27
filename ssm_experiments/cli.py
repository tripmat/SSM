import os
import json
import argparse
import sys
import io
from datetime import datetime
import atexit
import random
import shutil
import pickle
import numpy as np
import torch

from .models.registry import get_model
from .data import get_tokenizer, get_train_dataset, CopyDataset
from .train import train_model
from .evaluate import offline_accuracy_evaluation, evaluate_length_generalization, generate_fixed_eval_dataset
from .utils.parameter_matching import get_optimal_configs, count_parameters
from .utils.plotting import plot_all_models_comparison, plot_single_model_analysis
from .utils.config_loader import load_config


def main():
    # Suppress HuggingFace transformers warnings
    import warnings
    import logging

    # Suppress specific HuggingFace warnings
    warnings.filterwarnings("ignore", message=".*has generative capabilities.*")
    warnings.filterwarnings("ignore", message=".*GenerationMixin.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

    # Suppress PyTorch warnings
    warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

    # Set logging levels to reduce verbose output
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)

    # Set CUBLAS configuration for deterministic GPU training
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

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
    parser.add_argument('--config', type=str, default=None,
                        help='Path to Python config file (.py). If omitted, uses configs.py when present.')
    parser.add_argument('--only', type=str,
                        choices=['all', 'transformer', 'paper_mamba', 'minimal_mamba', 'transformer_rope', 'transformer_nope',
                                'transformer_alibi', 'transformer_hard_alibi'],
                        default='all',
                        help='Select which model(s) to run')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for training (auto=detect GPU, default: auto)')
    args_cli = parser.parse_args()

    # Device selection
    if args_cli.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args_cli.device

    # Validate device availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'

    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # Reproducibility: seed all relevant RNGs; allow override via env var
    seed = int(os.environ.get('SSM_SEED', '1337'))
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        print(f"Using seed: {seed}")
    except Exception as e:
        print(f"Warning: failed to set reproducible seeds: {e}")

    # Set up timestamped experiment directory and logging
    outputs_dir = os.path.abspath(args_cli.outputs)
    if os.path.islink(outputs_dir):
        raise ValueError(f"Refusing to write to symlinked outputs directory: {outputs_dir}")
    os.makedirs(outputs_dir, exist_ok=True)

    # Create timestamped experiment folder
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(outputs_dir, f'experiment_{ts}')
    logs_dir = os.path.join(experiment_dir, 'logs')
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    log_path = os.path.join(logs_dir, f'run_{ts}.log')
    print(f"Created experiment directory: {experiment_dir}")

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

    def evaluate_checkpoints_during_training(model_name, args, checkpoints_dir, tokenizer, TO_TOKEN, fixed_eval_dataset, device):
        """Evaluate all saved checkpoints to reconstruct training-time accuracy curve"""
        import glob

        # Find all checkpoint files for this model
        checkpoint_pattern = os.path.join(checkpoints_dir, f"{model_name}_step_*.pth")
        checkpoint_files = glob.glob(checkpoint_pattern)

        if not checkpoint_files:
            print(f"No checkpoints found for {model_name}")
            return [], []

        # Sort by step number
        def extract_step(filepath):
            try:
                filename = os.path.basename(filepath)
                # Extract step number from filename like "model_name_step_50.pth"
                step_part = filename.split('_step_')[1].split('.pth')[0]
                return int(step_part)
            except (IndexError, ValueError):
                return 0

        checkpoint_files.sort(key=extract_step)

        accuracies = []
        accuracy_training_examples = []

        print(f"Evaluating {len(checkpoint_files)} checkpoints for {model_name}...")

        for checkpoint_file in checkpoint_files:
            step = extract_step(checkpoint_file)

            try:
                # Create fresh model instance (suppress verbose output)
                import sys
                import io
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()  # Suppress get_model() prints
                try:
                    eval_model = get_model(args, tokenizer)
                finally:
                    sys.stdout = old_stdout

                # Load checkpoint weights
                try:
                    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
                except (TypeError, RuntimeError, pickle.UnpicklingError) as e:
                    # Fallback for comprehensive checkpoints containing RNG states
                    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)

                if not isinstance(checkpoint, dict):
                    raise ValueError("Unexpected checkpoint content")

                # Handle both old format (direct state_dict) and new format (comprehensive checkpoint)
                if 'model_state_dict' in checkpoint:
                    # New comprehensive format
                    model_state_dict = checkpoint['model_state_dict']

                    # Restore RNG states for deterministic evaluation
                    if 'torch_rng_state' in checkpoint:
                        torch.set_rng_state(checkpoint['torch_rng_state'])
                    if 'numpy_rng_state' in checkpoint:
                        np.random.set_state(checkpoint['numpy_rng_state'])
                    if 'random_rng_state' in checkpoint:
                        random.setstate(checkpoint['random_rng_state'])
                    if device == 'cuda':
                        if 'cuda_rng_state' in checkpoint:
                            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
                        if 'cuda_rng_state_all' in checkpoint:
                            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state_all'])
                else:
                    # Old format (direct state_dict) for backward compatibility
                    model_state_dict = checkpoint

                eval_model.load_state_dict(model_state_dict)
                eval_model = eval_model.to(device)
                eval_model.eval()

                # Evaluate accuracy (suppress verbose output)
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()  # Suppress evaluation prints
                try:
                    with torch.no_grad():
                        accuracy = offline_accuracy_evaluation(eval_model, fixed_eval_dataset, args, tokenizer, TO_TOKEN, device=device)
                finally:
                    sys.stdout = old_stdout

                accuracies.append(accuracy)
                # Convert step to training examples (step is 1-indexed, but step * batch_size gives examples after that step)
                accuracy_training_examples.append((step - 1) * args.train_batch_size)
                print(f"Evaluating {model_name}: Checkpoint at step {step}, Accuracy = {accuracy:.1f}%")

            except Exception as e:
                print(f"Warning: Failed to evaluate checkpoint at step {step}: {e}")
                accuracies.append(0.0)
                accuracy_training_examples.append((step - 1) * args.train_batch_size)

        return accuracies, accuracy_training_examples

    def load_compatible_results(figures_dir, current_config):
        """Load existing results that have compatible experiment configuration"""
        results_cache_path = os.path.join(figures_dir, 'current_experiment_results.json')
        config_cache_path = os.path.join(figures_dir, 'current_experiment_config.json')

        if not os.path.exists(results_cache_path) or not os.path.exists(config_cache_path):
            return {}, current_config

        try:
            # Load cached config
            with open(config_cache_path, 'r') as f:
                cached_config = json.load(f)

            # Load cached results
            with open(results_cache_path, 'r') as f:
                cached_results = json.load(f)

            # Check if configs are identical
            if cached_config == current_config:
                print(f"Found compatible previous results with {len(cached_results)} models")
                return cached_results, current_config
            else:
                print("Previous results found but configuration doesn't match - starting fresh")
                return {}, current_config

        except Exception as e:
            print(f"Warning: Failed to load cached results: {e}")
            return {}, current_config

    def save_experiment_results(figures_dir, results, experiment_config):
        """Save current experiment results and config for future compatibility checking"""
        results_cache_path = os.path.join(figures_dir, 'current_experiment_results.json')
        config_cache_path = os.path.join(figures_dir, 'current_experiment_config.json')

        # Convert results to serializable format
        serializable_results = {}
        for name, r in results.items():
            if r is not None:
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

        try:
            # Save results
            with open(results_cache_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)

            # Save config separately for easy comparison
            with open(config_cache_path, 'w') as f:
                json.dump(experiment_config, f, indent=2)

        except Exception as e:
            print(f"Warning: Failed to save experiment cache: {e}")

    def check_existing_results(model_name, args, checkpoints_dir):
        """Check if a model has already been trained to completion"""
        config_str = f"{model_name}_{args.hidden_size}h_{args.layers}l_{args.lr}lr_{args.steps}s"
        results_path = os.path.join(checkpoints_dir, f"{config_str}_results.json")

        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)
                # Check if results are complete
                if ('training' in results and 'fixed_accuracy' in results and
                    'length_gen' in results and len(results['training']['losses']) >= args.steps):
                    print(f"Found complete results for {model_name}, skipping training")
                    return results
            except Exception as e:
                print(f"Warning: Failed to load existing results for {model_name}: {e}")

        return None

    def validate_context_lengths(train_ctx, max_train_len, eval_ctx, max_eval_len):
        """Validate sequence lengths once with clear guidance.

        Requirements:
        - Training: context_len must be > 2 * max_train_len
        - Fixed-eval set (used for accuracy snapshots): eval_context_len must be > 2 * max_train_len
        - Length generalization: eval lengths beyond eval_context_len are skipped (FYI)
        """
        train_req = 2 * max_train_len
        eval_req_for_fixed = 2 * max_train_len

        problems = []
        if train_ctx <= train_req:
            problems.append(
                f"Training context_len={train_ctx} is too small for max_train_len={max_train_len} (needs > {train_req})."
            )
        if eval_ctx <= eval_req_for_fixed:
            problems.append(
                f"Eval context_len={eval_ctx} is too small to build the fixed eval set at length {max_train_len} (needs > {eval_req_for_fixed})."
            )

        if problems:
            print("\nConfiguration error: context lengths are insufficient for chosen string lengths.\n")
            for p in problems:
                print(f" - {p}")
            print(
                "\nRecommendations:\n"
                f" - Set --context-len to a value > {train_req} (e.g., {train_req + 20}).\n"
                f" - Set --eval-context-len to a value > {eval_req_for_fixed} (e.g., {eval_req_for_fixed + 50}).\n"
                " - Or reduce --max-train-len / --max-eval-len accordingly.\n"
            )
            raise SystemExit(2)

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

    # Optional: load user config (.py) to override globals and per-model params
    user_config = None
    # Auto-detect default config if not provided
    if not args_cli.config:
        default_cfg = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'configs.py'))
        if os.path.exists(default_cfg):
            args_cli.config = default_cfg
    if args_cli.config:
        cfg_path = os.path.abspath(args_cli.config)
        if not os.path.exists(cfg_path):
            raise SystemExit(f"Config file not found: {cfg_path}")
        try:
            user_config = load_config(cfg_path)
            print(f"Loaded config: {cfg_path}")
            # Config snapshots will be saved later to experiment directory
        except Exception as e:
            raise SystemExit(f"Failed to load config {cfg_path}: {e}")

    def apply_global_overrides(argobj):
        if not user_config:
            return argobj
        global_cfg = user_config.get('global') or user_config.get('globals')
        if not isinstance(global_cfg, dict):
            return argobj
        # Only override known keys to avoid typos silently passing
        allowed = {
            'steps', 'lr', 'max_lr', 'min_lr', 'warmup_steps', 'scheduler',
            'train_batch_size', 'eval_batch_size', 'eval_num_batches',
            'min_train_len', 'max_train_len', 'min_eval_len', 'max_eval_len',
            'context_len', 'eval_context_len', 'vocab_size', 'n_gram', 'length_answer'
        }
        for k, v in global_cfg.items():
            if k in allowed:
                setattr(argobj, k, v)
        return argobj

    # Setup outputs directories early (needed for config matching)
    checkpoints_dir = os.path.join(outputs_dir, 'checkpoints')
    figures_dir = os.path.join(experiment_dir, 'figures')

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Copy config file to experiment directory
    if user_config and hasattr(args_cli, 'config') and args_cli.config:
        import shutil
        config_backup_path = os.path.join(experiment_dir, 'config_used.py')
        try:
            shutil.copy2(args_cli.config, config_backup_path)
            print(f"Config backup saved: {config_backup_path}")
        except Exception as e:
            print(f"Warning: Could not backup config file: {e}")

    # Always save the resolved config as JSON for reproducibility
    if user_config:
        try:
            import json
            config_json_path = os.path.join(experiment_dir, 'config_resolved.json')
            with open(config_json_path, 'w') as f:
                json.dump(user_config, f, indent=2)
            print(f"Resolved config saved: {config_json_path}")
        except Exception as e:
            print(f"Warning: Could not save resolved config: {e}")

    # Setup experiment config for matching - use the full user_config or create default
    if user_config:
        experiment_config = user_config.copy()  # Use the loaded config
        print("Using provided Python config for compatibility matching")
    else:
        experiment_config = {"note": "no_config_file_provided", "args_cli": vars(args_cli)}
        print("No config file provided - using CLI arguments for compatibility matching")

    # Start fresh for each experiment (no more caching between runs)
    results = {}

    # If user provided target params and/or constraints, pass them through to matching
    target_params = None
    transformer_heads = None
    mamba_state_dim = None
    hard_num_masked_heads = None
    if user_config:
        g = user_config.get('global') or user_config.get('globals') or {}
        target_params = g.get('target_params')
        # Transformer heads: use any provided transformer variant heads (rope preferred)
        mcfg = user_config.get('models') or {}
        tr = (mcfg.get('transformer_rope') or mcfg.get('transformer') or {})
        if isinstance(tr, dict) and 'heads' in tr:
            transformer_heads = tr['heads']
        tha = mcfg.get('transformer_hard_alibi') or {}
        if isinstance(tha, dict) and 'num_masked_heads' in tha:
            hard_num_masked_heads = tha['num_masked_heads']
        pm = mcfg.get('paper_mamba') or {}
        # Use paper_mamba state_dim if available
        if isinstance(pm, dict) and 'state_dim' in pm:
            mamba_state_dim = pm['state_dim']

    # Determine vocab_size for matching (prefer config override if any)
    vocab_for_matching = None
    if user_config:
        g = user_config.get('global') or user_config.get('globals') or {}
        if isinstance(g, dict) and 'vocab_size' in g:
            vocab_for_matching = g['vocab_size']
    if vocab_for_matching is None:
        vocab_for_matching = getattr(Args(), 'vocab_size', 26)

    configs = get_optimal_configs(
        target_params=target_params or 10_000_000,
        vocab_size=vocab_for_matching,
        transformer_heads=transformer_heads,
        mamba_state_dim=mamba_state_dim,
        hard_num_masked_heads=hard_num_masked_heads or 6,
    )
    # Apply per-model overrides from config if provided
    if user_config and isinstance(user_config.get('models'), dict):
        user_models = user_config['models']
        for name, overrides in user_models.items():
            if not isinstance(overrides, dict):
                continue

            # If model is not in auto-generated configs, create it from configs.py
            if name not in configs:
                # For config-only models, estimate parameters based on target budget
                target = target_params or 5_000_000
                vocab_size = vocab_for_matching

                # More accurate estimation for model size based on type
                if name == 'minimal_mamba':
                    state_dim = overrides.get('state_dim', 16)
                    expand = overrides.get('expand', 2)

                    # Check if layers is fixed in config
                    if 'layers' in overrides:
                        # Fixed layers: find optimal hidden_size
                        from .utils.parameter_matching import estimate_minimal_mamba_params
                        layers = overrides['layers']
                        best_hidden = None
                        best_diff = float('inf')

                        for hidden in range(60, 200):
                            params = estimate_minimal_mamba_params(hidden, layers, state_dim, vocab_size, expand)
                            diff = abs(params - target)
                            if diff < best_diff and params <= target * 1.05:
                                best_hidden = hidden
                                best_diff = diff
                                best_params = params

                        if best_hidden:
                            hidden_size = best_hidden
                            print(f"Matched minimal_mamba: {hidden_size}h x {layers}L = {best_params:,} params (target: {target:,})")
                        else:
                            hidden_size, layers = 160, 6
                    else:
                        # Use parameter matching for both hidden_size and layers
                        from .utils.parameter_matching import find_minimal_mamba_config
                        config_result = find_minimal_mamba_config(
                            target_params=target,
                            vocab_size=vocab_size,
                            state_dim=state_dim,
                            expand=expand
                        )

                        if config_result:
                            hidden_size = config_result['hidden_size']
                            layers = config_result['layers']
                            print(f"Matched minimal_mamba: {hidden_size}h x {layers}L = {config_result['estimated_params']:,} params (target: {target:,})")
                        else:
                            hidden_size, layers = 384, 6
                elif name == 'paper_mamba':
                    state_dim = overrides.get('state_dim', 32)

                    # Check if layers is fixed in config
                    if 'layers' in overrides:
                        # Fixed layers: find optimal hidden_size
                        from .utils.parameter_matching import estimate_paper_mamba_params
                        layers = overrides['layers']
                        best_hidden = None
                        best_diff = float('inf')

                        for hidden in range(80, 160):
                            params = estimate_paper_mamba_params(hidden, layers, state_dim, vocab_size)
                            diff = abs(params - target)
                            if diff < best_diff and params <= target * 1.05:
                                best_hidden = hidden
                                best_diff = diff
                                best_params = params

                        if best_hidden:
                            hidden_size = best_hidden
                            print(f"Matched paper_mamba: {hidden_size}h x {layers}L = {best_params:,} params (target: {target:,})")
                        else:
                            hidden_size, layers = 160, 6
                    else:
                        # Use parameter matching for both hidden_size and layers
                        from .utils.parameter_matching import find_paper_mamba_config
                        config_result = find_paper_mamba_config(
                            target_params=target,
                            vocab_size=vocab_size,
                            state_dim=state_dim
                        )

                        if config_result:
                            hidden_size = config_result['hidden_size']
                            layers = config_result['layers']
                            print(f"Matched paper_mamba: {hidden_size}h x {layers}L = {config_result['estimated_params']:,} params (target: {target:,})")
                        else:
                            hidden_size, layers = 384, 6
                elif 'mamba' in name.lower():
                    # Other Mamba-like models - rough estimate
                    hidden_size = int((target / (vocab_size * 2 + 12 * 8)) ** 0.5)
                    hidden_size = max(128, min(1024, hidden_size))
                    layers = min(12, max(4, target // (hidden_size * hidden_size * 6)))
                else:
                    # Transformer-like models
                    hidden_size = int((target / (vocab_size * 2 + 12 * 8)) ** 0.5)
                    hidden_size = max(128, min(1024, hidden_size))
                    layers = min(12, max(4, target // (hidden_size * hidden_size * 12)))

                base_config = {
                    'model': name,
                    'hidden_size': hidden_size,
                    'layers': layers,
                    'heads': max(4, hidden_size // 64),  # reasonable head count
                }
                configs[name] = base_config

            # Apply all overrides
            for key in ('model', 'hidden_size', 'layers', 'heads', 'state_dim', 'dt_min', 'dt_max', 'expand', 'd_conv', 'num_masked_heads', 'dropout_rate'):
                if key in overrides:
                    configs[name][key] = overrides[key]
    # Optional filtering to run specific models
    if args_cli.only == 'transformer':
        configs = {k: v for k, v in configs.items() if k.startswith('transformer')}
    elif args_cli.only == 'all':
        # Run all available models (paper_mamba + transformers)
        pass  # Keep all configs as-is
    else:
        # Run specific model only
        configs = {k: v for k, v in configs.items() if k == args_cli.only}

    args_temp = Args()
    args_temp = apply_global_overrides(args_temp)
    # One-time validation for context vs lengths before any data/model work
    validate_context_lengths(
        train_ctx=args_temp.context_len,
        max_train_len=args_temp.max_train_len,
        eval_ctx=args_temp.eval_context_len,
        max_eval_len=args_temp.max_eval_len,
    )
    tokenizer, TO_TOKEN, TO_CHAR = get_tokenizer(args_temp)
    fixed_eval_dataset = generate_fixed_eval_dataset(args_temp, tokenizer, TO_TOKEN, num_examples=100)

    print("Starting experiments...")
    print("=" * 60)

    # Outputs directories already created earlier

    for model_name, config in configs.items():
        print(f"\nProcessing {model_name.upper()}...")
        print("-" * 40)

        args = Args()
        args = apply_global_overrides(args)
        args.model = config['model']
        args.hidden_size = config['hidden_size']
        args.layers = config['layers']
        if 'heads' in config:
            args.heads = config['heads']
        if 'state_dim' in config:
            args.state_dim = config['state_dim']
        if 'num_masked_heads' in config:
            args.num_masked_heads = config['num_masked_heads']
        if 'dt_min' in config:
            args.dt_min = config['dt_min']
        if 'dt_max' in config:
            args.dt_max = config['dt_max']
        if 'expand' in config:
            args.expand = config['expand']
        if 'd_conv' in config:
            args.d_conv = config['d_conv']
        if 'dropout_rate' in config:
            args.dropout_rate = config['dropout_rate']

        # Check if model was already trained to completion
        existing_results = check_existing_results(model_name, args, checkpoints_dir)
        if existing_results is not None:
            results[model_name] = existing_results
            print(f"Using existing results for {model_name}")

            # Still log comprehensive info for existing models
            model = get_model(args, tokenizer)
            from .utils.model_logger import log_comprehensive_model_info
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{model_name}_existing"
            log_data, log_path = log_comprehensive_model_info(
                model=model,
                model_name=model_name,
                args=args,
                tokenizer=tokenizer,
                output_dir=logs_dir,
                run_id=run_id
            )
            continue

        print(f"Training {model_name.upper()}...")
        # Print resolved LR schedule parameters
        max_lr = getattr(args, 'max_lr', getattr(args, 'lr', 1e-4))
        min_lr = getattr(args, 'min_lr', 1e-6)
        warmup_steps = getattr(args, 'warmup_steps', 300)
        print(f"LR schedule: warmup_steps={warmup_steps}, max_lr={max_lr}, min_lr={min_lr}")

        model = get_model(args, tokenizer)
        param_count = count_parameters(model)

        # Comprehensive model logging
        from .utils.model_logger import log_comprehensive_model_info
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{model_name}"
        log_data, log_path = log_comprehensive_model_info(
            model=model,
            model_name=model_name,
            args=args,
            tokenizer=tokenizer,
            output_dir=logs_dir,
            run_id=run_id
        )

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
        training_results = train_model(
            args,
            model,
            train_dataset,
            TO_TOKEN,
            tokenizer,
            device=device,
            checkpoint_dir=checkpoints_dir,
            model_name=model_name,
        )

        # Save final model checkpoint BEFORE evaluation, then reload for eval to ensure parity
        config_str = f"{model_name}_{args.hidden_size}h_{args.layers}l_{args.lr}lr_{args.steps}s"
        checkpoint_path = os.path.join(checkpoints_dir, f"{config_str}.pth")
        results_path = os.path.join(checkpoints_dir, f"{config_str}_results.json")

        try:
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved final checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Warning: failed to save model checkpoint: {e}")

        # Evaluate all checkpoints to reconstruct training-time accuracy curve
        print("Evaluating checkpoints from training...")
        checkpoint_accuracies, checkpoint_accuracy_examples = evaluate_checkpoints_during_training(
            model_name, args, checkpoints_dir, tokenizer, TO_TOKEN, fixed_eval_dataset, device
        )

        # Add checkpoint accuracies to training results
        training_results['accuracies'] = checkpoint_accuracies
        training_results['accuracy_training_examples'] = checkpoint_accuracy_examples

        # Recreate model and load saved weights for evaluation (parity with post-run evals)
        try:
            eval_model = get_model(args, tokenizer)
            # Safe load: prefer weights_only when available; validate type
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)  # type: ignore[arg-type]
            except (TypeError, RuntimeError, pickle.UnpicklingError):
                # Fallback for comprehensive checkpoints containing RNG states
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            if not isinstance(checkpoint, dict):
                raise ValueError("Unexpected checkpoint content (expected a dict)")

            # Handle both old format (direct state_dict) and new format (comprehensive checkpoint)
            if 'model_state_dict' in checkpoint:
                # New comprehensive format
                model_state_dict = checkpoint['model_state_dict']
            else:
                # Old format (direct state_dict) - check if it looks like a state_dict
                if all(hasattr(v, 'size') for v in checkpoint.values() if torch.is_tensor(v)):
                    model_state_dict = checkpoint
                else:
                    raise ValueError("Unexpected checkpoint content (not a valid state_dict)")

            eval_model.load_state_dict(model_state_dict)
            print("Reloaded model from final checkpoint for evaluation.")
        except Exception as e:
            print(f"Warning: failed to reload model from checkpoint: {e}. Using in-memory model for eval.")
            eval_model = model

        # Evaluate using reloaded model
        fixed_accuracy = offline_accuracy_evaluation(eval_model, fixed_eval_dataset, args, tokenizer, TO_TOKEN, device=device)
        length_gen_results = evaluate_length_generalization(args, eval_model, tokenizer, TO_TOKEN, device=device)

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

        # Generate updated comparison plot after each model
        print(f"Updating comparison plot with {len(results)} model(s)...")
        plot_all_models_comparison(results, os.path.join(figures_dir, 'all_models_comparison.png'))

        # Generate detailed single model analysis with overfitting progression
        print(f"Generating detailed analysis for {model_name}...")
        single_model_path = os.path.join(figures_dir, f'{model_name}_detailed_analysis.png')
        plot_single_model_analysis(model_name, results_obj, checkpoints_dir, args, tokenizer, TO_TOKEN, device, single_model_path)

        # Results will be saved at the end to experiment directory

    # Final experiment summary
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETED")
    print(f"{'='*60}")
    print(f"Models trained: {len(results)}")
    print(f"Final comparison plot: {os.path.join(figures_dir, 'all_models_comparison.png')}")

    # Save final detailed results to experiment directory
    results_file = os.path.join(experiment_dir, "experiment_results.json")
    with open(results_file, 'w') as f:
        serializable_results = {}
        for name, r in results.items():
            if r is not None:
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
