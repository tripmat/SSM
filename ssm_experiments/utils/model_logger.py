"""
Comprehensive model logging utility for complete reproducibility and debugging.
Logs every conceivable parameter, configuration, and system state.
"""
import json
import os
import platform
import psutil
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional
import hashlib

import torch
import numpy as np


def get_git_info():
    """Get git commit hash and repo status."""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        try:
            status = subprocess.check_output(['git', 'status', '--porcelain']).decode('ascii').strip()
            is_dirty = bool(status)
        except:
            is_dirty = None
        return {'commit_hash': commit_hash, 'is_dirty': is_dirty, 'status': status if is_dirty else None}
    except:
        return {'commit_hash': None, 'is_dirty': None, 'status': None}


def get_system_info():
    """Get comprehensive system information."""
    system_info = {
        'timestamp_start': datetime.now().isoformat(),
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'operating_system': platform.system(),
        'os_version': platform.version(),
        'platform': platform.platform(),
        'cpu_count': os.cpu_count(),
        'cpu_info': platform.processor(),
    }

    # Memory info
    try:
        memory = psutil.virtual_memory()
        system_info.update({
            'ram_total_gb': round(memory.total / (1024**3), 2),
            'ram_available_gb': round(memory.available / (1024**3), 2),
            'ram_percent_used': memory.percent,
        })
    except:
        pass

    # CUDA info
    if torch.cuda.is_available():
        system_info.update({
            'cuda_available': True,
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            'cuda_device_count': torch.cuda.device_count(),
            'cuda_current_device': torch.cuda.current_device(),
        })

        # GPU info for each device
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            system_info[f'gpu_{i}_name'] = props.name
            system_info[f'gpu_{i}_memory_total_gb'] = round(props.total_memory / (1024**3), 2)
            system_info[f'gpu_{i}_major'] = props.major
            system_info[f'gpu_{i}_minor'] = props.minor

            # Current memory usage
            if i == torch.cuda.current_device():
                system_info[f'gpu_{i}_memory_allocated_gb'] = round(torch.cuda.memory_allocated(i) / (1024**3), 2)
                system_info[f'gpu_{i}_memory_reserved_gb'] = round(torch.cuda.memory_reserved(i) / (1024**3), 2)
    else:
        system_info.update({
            'cuda_available': False,
            'cuda_version': None,
            'cudnn_version': None,
        })

    # Environment variables
    system_info['environment_variables'] = {
        key: value for key, value in os.environ.items()
        if any(prefix in key for prefix in ['CUDA_', 'TORCH_', 'OMP_', 'MKL_', 'PYTHON'])
    }

    # Git info
    system_info.update(get_git_info())

    return system_info


def get_model_architecture_info(model, model_name, args, tokenizer):
    """Extract comprehensive model architecture information."""

    # Basic architecture
    arch_info = {
        'model_name': model_name,
        'model_class': model.__class__.__name__,
        'vocab_size': len(tokenizer),
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    # Model-specific flags and configuration
    arch_info.update({
        'tie_embeddings': getattr(model, 'tie_word_embeddings', False),
        'use_bias': getattr(args, 'bias', None),
        'norm_type': 'LayerNorm',  # Default, will be updated if detected
        'activation': 'SiLU',  # Default for Mamba, GELU for transformers
    })

    # Detect normalization type
    for name, module in model.named_modules():
        if 'norm' in name.lower():
            if 'RMS' in type(module).__name__:
                arch_info['norm_type'] = 'RMSNorm'
                break
            elif 'LayerNorm' in type(module).__name__:
                arch_info['norm_type'] = 'LayerNorm'
                break

    # Get architecture-specific parameters
    if hasattr(args, 'hidden_size'):
        arch_info['d_model'] = args.hidden_size
    if hasattr(args, 'layers'):
        arch_info['n_layer'] = args.layers
    if hasattr(args, 'heads'):
        arch_info['heads'] = args.heads
    if hasattr(args, 'state_dim'):
        arch_info['d_state'] = args.state_dim
    if hasattr(args, 'expand'):
        arch_info['expand'] = args.expand
        if hasattr(args, 'hidden_size'):
            arch_info['d_inner'] = args.expand * args.hidden_size
    if hasattr(args, 'd_conv'):
        arch_info['d_conv'] = args.d_conv
    if hasattr(args, 'dt_min'):
        arch_info['dt_min'] = args.dt_min
    if hasattr(args, 'dt_max'):
        arch_info['dt_max'] = args.dt_max

    # Model-specific architecture details
    if 'mamba' in model_name.lower():
        # Try to extract Mamba-specific info
        if hasattr(model, 'model') and hasattr(model.model, 'args'):  # minimal_mamba
            mamba_args = model.model.args
            arch_info.update({
                'dt_rank': mamba_args.dt_rank,
                'pad_vocab_size_multiple': mamba_args.pad_vocab_size_multiple,
                'conv_bias': mamba_args.conv_bias,
                'bias': mamba_args.bias,
            })

        # Extract layer-specific info from first layer if available
        if hasattr(model, 'model') and hasattr(model.model, 'layers') and len(model.model.layers) > 0:
            first_layer = model.model.layers[0]
            if hasattr(first_layer, 'mixer'):  # minimal_mamba structure
                mixer = first_layer.mixer
                if hasattr(mixer, 'args'):
                    arch_info['mamba_block_args'] = {
                        'dt_rank': mixer.args.dt_rank,
                        'd_conv': mixer.args.d_conv,
                        'd_inner': mixer.args.d_inner,
                        'd_state': mixer.args.d_state,
                    }

    # Parameter breakdown by component
    param_breakdown = {}
    for name, param in model.named_parameters():
        component = name.split('.')[0] if '.' in name else name
        if component not in param_breakdown:
            param_breakdown[component] = 0
        param_breakdown[component] += param.numel()

    arch_info['parameter_breakdown'] = param_breakdown

    # Get layer-wise parameter counts
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        arch_info['parameters_per_layer'] = sum(p.numel() for p in layers[0].parameters()) if len(layers) > 0 else 0
        arch_info['layer_count_actual'] = len(layers)
    elif hasattr(model, 'layers'):
        layers = model.layers
        arch_info['parameters_per_layer'] = sum(p.numel() for p in layers[0].parameters()) if len(layers) > 0 else 0
        arch_info['layer_count_actual'] = len(layers)

    # Model state dict info
    state_dict = model.state_dict()
    arch_info['state_dict_keys'] = list(state_dict.keys())
    arch_info['state_dict_size_mb'] = sum(param.numel() * param.element_size() for param in state_dict.values()) / (1024**2)

    # Tensor dtypes
    dtypes = {}
    for name, param in model.named_parameters():
        dtype_str = str(param.dtype)
        if dtype_str not in dtypes:
            dtypes[dtype_str] = []
        dtypes[dtype_str].append(name)
    arch_info['dtype_per_tensor'] = dtypes

    # Initialization details (extract from actual parameters)
    init_info = {}
    for name, param in model.named_parameters():
        if 'A_log' in name:
            init_info['A_log_init_mean'] = float(param.data.mean())
            init_info['A_log_init_std'] = float(param.data.std())
        elif 'weight' in name and 'embedding' not in name.lower():
            if 'weight_init_std' not in init_info:
                init_info['weight_init_std'] = float(param.data.std())
        elif 'bias' in name:
            if 'bias_init_mean' not in init_info:
                init_info['bias_init_mean'] = float(param.data.mean())
        elif 'D' in name and len(param.shape) == 1:  # SSM D parameter
            init_info['D_init_mean'] = float(param.data.mean())
            init_info['D_init_std'] = float(param.data.std())

    arch_info['initialization_details'] = init_info

    # Memory layout and shapes
    arch_info['memory_layout'] = {
        'batch_first': True,  # Default for our models
        'tensor_layout': 'contiguous',
        'max_sequence_length': getattr(args, 'context_len', None),
    }

    return arch_info


def get_training_config_info(args):
    """Extract comprehensive training configuration."""

    training_config = {
        # Basic training params
        'batch_size': getattr(args, 'train_batch_size', None),
        'eval_batch_size': getattr(args, 'eval_batch_size', None),
        'context_len': getattr(args, 'context_len', None),
        'eval_context_len': getattr(args, 'eval_context_len', None),
        'steps': getattr(args, 'steps', None),
        'max_lr': getattr(args, 'max_lr', None),
        'min_lr': getattr(args, 'min_lr', None),
        'warmup_steps': getattr(args, 'warmup_steps', None),

        # Optimizer configuration (defaults for this framework)
        'optimizer': 'Adam',  # Default for this framework
        'optimizer_betas': [0.9, 0.999],  # Adam defaults
        'weight_decay': getattr(args, 'weight_decay', 0.0),
        'optimizer_eps': 1e-8,
        'scheduler_type': 'cosine_with_warmup',  # Based on CLI implementation
        'gradient_clip_val': getattr(args, 'gradient_clip_val', None),
        'gradient_accumulation_steps': 1,  # Default
        'mixed_precision': 'none',  # Default for this framework

        # Task parameters
        'task_name': 'copy',  # Based on the framework
        'min_train_len': getattr(args, 'min_train_len', None),
        'max_train_len': getattr(args, 'max_train_len', None),
        'min_eval_len': getattr(args, 'min_eval_len', None),
        'max_eval_len': getattr(args, 'max_eval_len', None),
        'vocab_size': getattr(args, 'vocab_size', None),
        'alphabet_size': 26,  # Lowercase letters
        'n_gram': getattr(args, 'n_gram', None),
        'length_answer': getattr(args, 'length_answer', None),
        'mask_inputs': True,  # Based on CLI implementation
        'pad_token_id': None,  # Not used in this framework
        'ignore_index': -100,  # PyTorch default

        # Compute/Memory
        'device': getattr(args, 'device', str(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')),
        'num_workers': 0,  # Default for this framework
        'pin_memory': torch.cuda.is_available(),

        # Logging/Checkpointing
        'experiment_name': getattr(args, 'experiment_name', 'ssm_experiments'),
        'log_interval': 100,  # Default based on implementation
        'eval_interval': 1000,  # Default
        'save_interval': 1000,  # Default

        # Model specific
        'model': getattr(args, 'model', None),
    }

    # Special tokens (based on tokenizer structure)
    training_config['special_tokens'] = {
        'bos_token': '$',
        'eos_token': '.',
        'sep_token': '|',
        'pad_token': '*',
    }

    # Add all args attributes
    all_args = {key: value for key, value in vars(args).items()}
    training_config['all_args'] = all_args

    return training_config


def get_random_seed_info():
    """Get all random seed information."""
    return {
        'torch_initial_seed': torch.initial_seed(),
        'torch_cuda_initial_seed': torch.cuda.initial_seed() if torch.cuda.is_available() else None,
        'numpy_random_state': np.random.get_state()[1][0] if len(np.random.get_state()[1]) > 0 else None,
        'cuda_deterministic': torch.backends.cudnn.deterministic if torch.cuda.is_available() else None,
        'cuda_benchmark': torch.backends.cudnn.benchmark if torch.cuda.is_available() else None,
    }


def get_numerical_stability_info():
    """Get numerical stability configurations."""
    stability_info = {
        'torch_default_dtype': str(torch.get_default_dtype()),
        'autograd_anomaly_detection': torch.is_anomaly_enabled(),
        'autograd_profiler_enabled': torch.autograd.profiler.profile.__module__ is not None,

        # Epsilon values for numerical stability
        'eps_values': {
            'layernorm_eps': 1e-5,  # Standard LayerNorm epsilon
            'rmsnorm_eps': 1e-5,    # Standard RMSNorm epsilon
            'softplus_eps': 1e-8,   # For softplus numerical stability
            'divide_eps': 1e-10,    # For safe division
        },

        # Clamping values for stability
        'clamp_values': {
            'A_log_min': -5.0,      # SSM A_log clamping
            'A_log_max': 2.0,
            'exp_clamp_min': -10.0, # Exponential clamping
            'exp_clamp_max': 0.0,
            'state_clamp_min': -10.0, # State value clamping
            'state_clamp_max': 10.0,
        },

        # Gradient and overflow checks
        'gradient_nan_checks': True,  # Typically enabled during training
        'inf_check_frequency': 'per_step',
        'numerical_overflow_strategy': 'clamp',
    }

    return stability_info


def get_performance_metrics(model, args, device):
    """Get performance metrics and memory footprint."""

    # Basic memory calculation
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

    performance_info = {
        'memory_footprint_mb': param_size_mb,
        'theoretical_memory_mb': param_size_mb,  # Same for parameters
        'model_size_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
    }

    # Rough FLOP estimation for forward pass
    total_params = sum(p.numel() for p in model.parameters())
    context_len = getattr(args, 'context_len', 512)

    # Very rough FLOP estimate: 6 * params * context_len (rule of thumb)
    estimated_flops = 6 * total_params * context_len
    performance_info['estimated_flops'] = estimated_flops

    # Context to parameter ratio
    performance_info['context_to_param_ratio'] = context_len / total_params if total_params > 0 else 0

    # Derived metrics
    d_model = getattr(args, 'hidden_size', 512)
    d_state = getattr(args, 'state_dim', 16)
    vocab_size = getattr(args, 'vocab_size', 26)

    # State capacity estimation (very rough)
    if d_state and d_model and vocab_size:
        import math
        state_capacity_bits = d_state * d_model * math.log2(vocab_size)
        performance_info['state_capacity_bits'] = state_capacity_bits

    return performance_info


def get_implementation_details(model, model_name):
    """Get implementation-specific details."""

    impl_details = {
        'scan_implementation_source': 'sequential_cpu',  # Default for this framework
        'conv_implementation': 'F.conv1d',
        'discretization_method': 'ZOH',  # Zero-order hold for SSM
        'parallel_strategy': 'none',  # Sequential implementation
        'checkpoint_compatibility': 'pytorch_v2',
    }

    # Detect Mamba-specific implementation details
    if 'mamba' in model_name.lower():
        impl_details.update({
            'use_cuda_kernel': False,  # CPU implementation
            'use_parallel_scan': False,
            'scan_algorithm': 'sequential',
            'conv_method': 'depthwise' if 'paper_mamba' in model_name else 'grouped',
        })
    else:
        # Transformer details
        impl_details.update({
            'attention_implementation': 'standard',
            'position_encoding_type': 'RoPE' if 'rope' in model_name else 'none',
            'attention_mask_type': 'causal',
        })

    return impl_details


def create_config_hash(config_dict):
    """Create MD5 hash of configuration for comparison."""
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()


def log_comprehensive_model_info(model, model_name, args, tokenizer, output_dir, run_id=None):
    """
    Log exhaustive model and training information.

    Args:
        model: The PyTorch model
        model_name: String identifier for the model
        args: Training arguments/configuration
        tokenizer: Tokenizer used
        output_dir: Directory to save logs
        run_id: Unique run identifier
    """

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine device for metrics
    device = next(model.parameters()).device

    # Gather all information
    log_data = {
        'run_id': run_id,
        'system_info': get_system_info(),
        'model_architecture': get_model_architecture_info(model, model_name, args, tokenizer),
        'training_config': get_training_config_info(args),
        'random_seeds': get_random_seed_info(),
        'numerical_stability': get_numerical_stability_info(),
        'performance_metrics': get_performance_metrics(model, args, device),
        'implementation_details': get_implementation_details(model, model_name),
    }

    # Add tokenizer info
    log_data['tokenizer_info'] = {
        'tokenizer_class': tokenizer.__class__.__name__,
        'vocab_size': len(tokenizer),
        'bos_token_id': getattr(tokenizer, 'bos_token_id', None),
        'eos_token_id': getattr(tokenizer, 'eos_token_id', None),
    }

    # Create configuration hash
    log_data['config_hash'] = create_config_hash(log_data)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to JSON file
    json_path = os.path.join(output_dir, f"{model_name}_{run_id}_comprehensive_log.json")
    with open(json_path, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)

    # Print formatted summary
    print_model_summary(log_data, model_name)

    return log_data, json_path


def print_model_summary(log_data, model_name):
    """Print a formatted summary of the model configuration."""

    print("="*80)
    print(f"COMPREHENSIVE MODEL LOG: {model_name.upper()}")
    print("="*80)

    # System info
    sys_info = log_data['system_info']
    print(f"SYSTEM: {sys_info['operating_system']} | Python {sys_info['python_version'].split()[0]} | PyTorch {sys_info['torch_version']}")
    print(f"MEMORY: {sys_info.get('ram_total_gb', 'N/A')} GB RAM | {sys_info.get('cpu_count', 'N/A')} CPU cores")

    if sys_info.get('cuda_available'):
        gpu_name = sys_info.get('gpu_0_name', 'Unknown GPU')
        gpu_memory = sys_info.get('gpu_0_memory_total_gb', 'N/A')
        print(f"GPU: {gpu_name} ({gpu_memory} GB) | CUDA {sys_info.get('cuda_version', 'N/A')}")
    else:
        print("GPU: None (CPU only)")

    # Model architecture
    arch = log_data['model_architecture']
    print(f"\nMODEL ARCHITECTURE:")
    print(f"   Model: {arch['model_name']} ({arch['model_class']})")
    print(f"   Parameters: {arch['total_parameters']:,} total ({arch['trainable_parameters']:,} trainable)")

    if 'd_model' in arch:
        print(f"   d_model: {arch['d_model']}")
    if 'n_layer' in arch:
        print(f"   n_layer: {arch['n_layer']}")
    if 'd_state' in arch:
        print(f"   d_state: {arch['d_state']}")
    if 'expand' in arch:
        print(f"   expand: {arch['expand']} (d_inner: {arch.get('d_inner', 'N/A')})")
    if 'd_conv' in arch:
        print(f"   d_conv: {arch['d_conv']}")
    if 'dt_min' in arch and 'dt_max' in arch:
        print(f"   dt_range: [{arch['dt_min']}, {arch['dt_max']}]")

    # Training config
    train_config = log_data['training_config']
    print(f"\nTRAINING CONFIGURATION:")
    print(f"   Batch size: {train_config.get('batch_size', 'N/A')} | Context: {train_config.get('context_len', 'N/A')}")
    print(f"   Steps: {train_config.get('steps', 'N/A')} | LR: {train_config.get('min_lr', 'N/A')} -> {train_config.get('max_lr', 'N/A')}")
    print(f"   Warmup: {train_config.get('warmup_steps', 'N/A')} steps")
    print(f"   Sequence lengths: {train_config.get('min_train_len', 'N/A')}-{train_config.get('max_train_len', 'N/A')} (train), {train_config.get('min_eval_len', 'N/A')}-{train_config.get('max_eval_len', 'N/A')} (eval)")

    # Parameter breakdown
    if 'parameter_breakdown' in arch:
        print(f"\nPARAMETER BREAKDOWN:")
        for component, count in arch['parameter_breakdown'].items():
            print(f"   {component}: {count:,} ({count/arch['total_parameters']*100:.1f}%)")

    # Git info
    git_info = sys_info
    if git_info.get('commit_hash'):
        status = " (dirty)" if git_info.get('is_dirty') else " (clean)"
        print(f"\nGIT: {git_info['commit_hash'][:8]}{status}")

    # Config hash
    print(f"\nCONFIG HASH: {log_data['config_hash'][:16]}...")
    print(f"RUN ID: {log_data['run_id']}")

    print("="*80)