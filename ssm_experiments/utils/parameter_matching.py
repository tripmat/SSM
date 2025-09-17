"""
Parameter matching utilities for fair model comparison
Ensures Transformer and Mamba models have similar parameter counts
"""

from . import __package__  # noqa: F401
from ..models.registry import get_model
from ..data.tokenizer import get_tokenizer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_transformer_params(hidden_size, layers, heads, vocab_size):
    embedding_params = vocab_size * hidden_size
    params_per_layer = 4 * hidden_size**2 + 8 * hidden_size**2 + 2 * hidden_size
    transformer_params = params_per_layer * layers
    output_params = hidden_size + vocab_size * hidden_size
    total_params = embedding_params + transformer_params + output_params
    return total_params


def estimate_mamba_params(hidden_size, layers, state_dim, vocab_size):
    embedding_params = vocab_size * hidden_size
    params_per_layer = (
        2 * hidden_size * hidden_size +  # Input/output projections
        2 * hidden_size * state_dim +    # SSM matrices
        hidden_size * 4 +                # Biases and small matrices
        hidden_size * hidden_size        # Residual connections
    )
    mamba_params = params_per_layer * layers
    output_params = vocab_size * hidden_size
    total_params = embedding_params + mamba_params + output_params
    return total_params


def find_matched_configs(target_params=5_000_000, vocab_size=30):
    best_configs = {}
    best_diff = float('inf')

    for hidden_size in [256, 320, 384, 448, 512]:
        for layers in [6, 8, 10, 12]:
            heads = max(4, hidden_size // 64)
            if hidden_size % heads != 0:
                continue
            est_params = estimate_transformer_params(hidden_size, layers, heads, vocab_size)
            if abs(est_params - target_params) < abs(target_params * 0.5):
                for mamba_layers in [8, 10, 12, 14, 16]:
                    for state_dim in [16, 24, 32, 48]:
                        mamba_params = estimate_mamba_params(hidden_size, mamba_layers, state_dim, vocab_size)
                        param_diff = abs(est_params - mamba_params)
                        if param_diff < best_diff:
                            best_diff = param_diff
                            best_configs = {
                                'transformer_rope': {
                                    'hidden_size': hidden_size,
                                    'layers': layers,
                                    'heads': heads,
                                    'estimated_params': est_params,
                                    'model': 'T_rope',
                                },
                                'mamba': {
                                    'hidden_size': hidden_size,
                                    'layers': mamba_layers,
                                    'state_dim': state_dim,
                                    'estimated_params': mamba_params,
                                    'model': 'mamba',
                                },
                                'param_difference': param_diff,
                                'param_ratio': mamba_params / est_params,
                            }
    return best_configs


def verify_actual_parameters(configs, vocab_size=30):
    class Args:
        def __init__(self):
            self.vocab_size = vocab_size
            self.train_task = "copy"
            self.eval_task = "copy"
            self.n_gram = 0
            self.length_answer = 0
            self.min_train_len = 10
            self.max_train_len = 50
            self.context_len = 200

    args = Args()
    tokenizer, _, _ = get_tokenizer(args)

    results = {}

    t_cfg = configs['transformer_rope']
    args.model = 'T_rope'
    args.hidden_size = t_cfg['hidden_size']
    args.layers = t_cfg['layers']
    args.heads = t_cfg['heads']
    transformer_model = get_model(args, tokenizer)
    transformer_params = count_parameters(transformer_model)
    results['transformer'] = {'config': t_cfg, 'actual_params': transformer_params, 'model': transformer_model}

    m_cfg = configs['mamba']
    args.model = 'mamba'
    args.hidden_size = m_cfg['hidden_size']
    args.layers = m_cfg['layers']
    args.state_dim = m_cfg['state_dim']
    mamba_model = get_model(args, tokenizer)
    mamba_params = count_parameters(mamba_model)
    results['mamba'] = {'config': m_cfg, 'actual_params': mamba_params, 'model': mamba_model}

    param_diff = abs(transformer_params - mamba_params)
    param_ratio = mamba_params / transformer_params
    results['comparison'] = {
        'param_difference': param_diff,
        'param_ratio': param_ratio,
        'average_params': (transformer_params + mamba_params) / 2,
    }
    return results


def get_optimal_configs():
    matched = find_matched_configs(target_params=5_000_000, vocab_size=30)
    verification = verify_actual_parameters(matched)
    final = {
        'transformer_rope': {
            'model': 'T_rope',
            'hidden_size': verification['transformer']['config']['hidden_size'],
            'layers': verification['transformer']['config']['layers'],
            'heads': verification['transformer']['config']['heads'],
            'actual_params': verification['transformer']['actual_params'],
        },
        'mamba': {
            'model': 'mamba',
            'hidden_size': verification['mamba']['config']['hidden_size'],
            'layers': verification['mamba']['config']['layers'],
            'state_dim': verification['mamba']['config']['state_dim'],
            'actual_params': verification['mamba']['actual_params'],
        },
    }
    return final

