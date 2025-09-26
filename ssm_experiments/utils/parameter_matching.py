"""
Parameter matching utilities for fair model comparison.
Caps all models to a target parameter budget by sweeping sizes and
matching Mamba to the Transformer baseline.
"""

from ..models.registry import get_model
from ..data.tokenizer import get_tokenizer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_transformer_params(hidden_size, layers, heads, vocab_size):
    embedding_params = vocab_size * hidden_size
    # Very rough per-layer estimate for GPT-NeoX block
    params_per_layer = 12 * hidden_size ** 2 + 2 * hidden_size
    transformer_params = params_per_layer * layers
    output_params = hidden_size + vocab_size * hidden_size
    total_params = embedding_params + transformer_params + output_params
    return total_params


def estimate_mamba_params(hidden_size, layers, state_dim, vocab_size):
    embedding_params = vocab_size * hidden_size
    params_per_layer = (
        2 * hidden_size * hidden_size +  # in/out projections
        2 * hidden_size * state_dim +    # SSM matrices
        hidden_size * 4 +                # biases/small matrices
        hidden_size * hidden_size        # residual/conv-like terms
    )
    mamba_params = params_per_layer * layers
    output_params = vocab_size * hidden_size
    total_params = embedding_params + mamba_params + output_params
    return total_params


def estimate_minimal_mamba_params(hidden_size, layers, state_dim, vocab_size, expand=2, dt_rank=None):
    """
    More accurate parameter estimation for minimal_mamba architecture.
    Based on the actual MinimalMamba implementation.
    """
    if dt_rank is None:
        dt_rank = max(1, hidden_size // 16)  # auto dt_rank calculation

    d_inner = int(expand * hidden_size)

    # Embedding and output head (tied weights)
    embedding_params = vocab_size * hidden_size

    # Per-layer parameters
    params_per_layer = (
        # in_proj: d_model -> d_inner * 2
        hidden_size * (d_inner * 2) +

        # conv1d: depthwise conv, d_inner channels, kernel_size=4
        d_inner * 4 +  # weights
        d_inner +      # bias

        # x_proj: d_inner -> dt_rank + state_dim * 2
        d_inner * (dt_rank + state_dim * 2) +

        # dt_proj: dt_rank -> d_inner
        dt_rank * d_inner + d_inner +  # weights + bias

        # A_log parameter: d_inner * state_dim
        d_inner * state_dim +

        # D parameter: d_inner
        d_inner +

        # out_proj: d_inner -> d_model
        d_inner * hidden_size +

        # RMSNorm: d_model
        hidden_size
    )

    total_layer_params = params_per_layer * layers

    # Final RMSNorm
    final_norm_params = hidden_size

    total_params = embedding_params + total_layer_params + final_norm_params
    return total_params


def find_minimal_mamba_config(target_params, vocab_size, state_dim, expand=2):
    """
    Find hidden_size and layers for minimal_mamba to match target parameter count.
    """
    best_config = None
    best_diff = float('inf')

    # Search reasonable ranges
    for hidden_size in range(128, 1024, 32):  # step by 32 for reasonable sizes
        for layers in range(2, 20):
            params = estimate_minimal_mamba_params(
                hidden_size, layers, state_dim, vocab_size, expand
            )
            diff = abs(params - target_params)

            if diff < best_diff:
                best_diff = diff
                best_config = {
                    'hidden_size': hidden_size,
                    'layers': layers,
                    'estimated_params': params,
                    'state_dim': state_dim,
                    'expand': expand
                }

            # Early stopping if we're very close
            if diff < target_params * 0.02:  # within 2%
                break
        else:
            continue
        break

    return best_config


def estimate_paper_mamba_params(hidden_size, layers, state_dim, vocab_size):
    """
    Accurate parameter estimation for paper_mamba (PaperInformedMambaBlock) architecture.
    Based on the actual PaperInformedMambaBlock implementation.
    """
    # Embedding (tied with output head)
    embedding_params = vocab_size * hidden_size

    # Per-layer parameters for PaperInformedMambaBlock
    params_per_layer = (
        # in_proj: d_model -> d_model * 2
        hidden_size * (hidden_size * 2) +

        # conv1d: depthwise conv, d_model channels, kernel_size=4, groups=d_model
        hidden_size * 4 +  # weights
        hidden_size +      # bias

        # A_log parameter: d_model * d_state
        hidden_size * state_dim +

        # B_proj: depthwise 1x1 conv, d_model -> d_model * d_state, groups=d_model
        (hidden_size * state_dim) +  # weights (grouped conv)
        (hidden_size * state_dim) +  # bias

        # C_proj: depthwise 1x1 conv, d_model -> d_model * d_state, groups=d_model
        (hidden_size * state_dim) +  # weights (grouped conv)
        (hidden_size * state_dim) +  # bias

        # dt_proj: d_model -> d_model
        hidden_size * hidden_size +

        # out_proj: d_model -> d_model
        hidden_size * hidden_size +

        # LayerNorm: weight + bias
        hidden_size * 2
    )

    total_layer_params = params_per_layer * layers

    total_params = embedding_params + total_layer_params
    return total_params


def find_paper_mamba_config(target_params, vocab_size, state_dim):
    """
    Find hidden_size and layers for paper_mamba to match target parameter count.
    """
    best_config = None
    best_diff = float('inf')

    # Search reasonable ranges
    for hidden_size in range(128, 1024, 32):  # step by 32 for reasonable sizes
        for layers in range(2, 20):
            params = estimate_paper_mamba_params(
                hidden_size, layers, state_dim, vocab_size
            )
            diff = abs(params - target_params)

            if diff < best_diff:
                best_diff = diff
                best_config = {
                    'hidden_size': hidden_size,
                    'layers': layers,
                    'estimated_params': params,
                    'state_dim': state_dim
                }

            # Early stopping if we're very close
            if diff < target_params * 0.02:  # within 2%
                break
        else:
            continue
        break

    return best_config


def find_matched_configs(
    target_params=10_000_000,
    vocab_size=30,
    transformer_heads=None,
    mamba_state_dim=None,
    hidden_sizes=None,
    transformer_layers=None,
    mamba_layers=None,
):
    """
    Search transformer + mamba sizes near a target parameter budget.
    Primary objective: minimize |Transformer params - target|.
    Secondary: minimize |Transformer params - Mamba params|.

    Optional constraints:
    - transformer_heads: fix number of heads used by transformer while searching.
    - mamba_state_dim: fix state dimension for Mamba while searching.
    - hidden_sizes, transformer_layers, mamba_layers: restrict candidate grids.
    """
    best_configs = {}
    best_key = (float('inf'), float('inf'))  # (target_error, pair_diff)

    hs_list = hidden_sizes or [256, 288, 320, 352, 384, 416, 448, 480, 512, 576]
    t_layers_list = transformer_layers or [4, 6, 8, 10, 12]
    m_layers_list = mamba_layers or [8, 12, 16, 20, 24, 28]
    sd_list = [mamba_state_dim] if mamba_state_dim is not None else [16, 24, 32, 48]

    for hidden_size in hs_list:
        for layers in t_layers_list:
            if transformer_heads is None:
                heads = max(6, hidden_size // 64)
            else:
                heads = transformer_heads
            if hidden_size % heads != 0 or heads <= 0:
                continue
            est_params = estimate_transformer_params(hidden_size, layers, heads, vocab_size)
            target_error = abs(est_params - target_params)
            # Only consider candidates not wildly off target
            if target_error < abs(target_params * 0.6):
                for mamba_layers in m_layers_list:
                    for state_dim in sd_list:
                        m_params = estimate_mamba_params(hidden_size, mamba_layers, state_dim, vocab_size)
                        pair_diff = abs(est_params - m_params)
                        key = (target_error, pair_diff)
                        if key < best_key:
                            best_key = key
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
                                    'estimated_params': m_params,
                                    'model': 'mamba',
                                },
                                'param_difference': pair_diff,
                                'param_ratio': m_params / est_params,
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
    results['transformer'] = {'config': t_cfg, 'actual_params': transformer_params}

    m_cfg = configs['mamba']
    args.model = 'mamba'
    args.hidden_size = m_cfg['hidden_size']
    args.layers = m_cfg['layers']
    args.state_dim = m_cfg['state_dim']
    mamba_model = get_model(args, tokenizer)
    mamba_params = count_parameters(mamba_model)
    results['mamba'] = {'config': m_cfg, 'actual_params': mamba_params}

    results['comparison'] = {
        'param_difference': abs(transformer_params - mamba_params),
        'param_ratio': mamba_params / transformer_params,
        'average_params': (transformer_params + mamba_params) / 2,
    }
    return results


def get_optimal_configs(
    target_params=10_000_000,
    vocab_size=30,
    transformer_heads=None,
    mamba_state_dim=None,
    hard_num_masked_heads=6,
):
    """
    Return configs matched to a target parameter budget with optional constraints.

    - Transformer (GPT-NeoX) baseline chosen to hit target params using given heads if provided.
    - Mamba matched to similar params using given state_dim if provided.
    - Hard-ALiBi masked heads set via hard_num_masked_heads (no param change).
    """
    matched = find_matched_configs(
        target_params=target_params,
        vocab_size=vocab_size,
        transformer_heads=transformer_heads,
        mamba_state_dim=mamba_state_dim,
    )
    verification = verify_actual_parameters(matched, vocab_size=vocab_size)

    base = verification['transformer']['config']
    base_hidden_size = base['hidden_size']
    base_layers = base['layers']
    base_heads = base['heads']

    final = {
        'transformer_rope': {
            'model': 'T_rope',
            'hidden_size': base_hidden_size,
            'layers': base_layers,
            'heads': base_heads,
            'actual_params': verification['transformer']['actual_params'],
        },
        'transformer_nope': {
            'model': 'T_nope',
            'hidden_size': base_hidden_size,
            'layers': base_layers,
            'heads': base_heads,
            'actual_params': verification['transformer']['actual_params'],
        },
        'transformer_alibi': {
            'model': 'T_alibi',
            'hidden_size': base_hidden_size,
            'layers': base_layers,
            'heads': base_heads,
            'actual_params': verification['transformer']['actual_params'],
        },
        'transformer_hard_alibi': {
            'model': 'T_hard_alibi',
            'hidden_size': base_hidden_size,
            'layers': base_layers,
            'heads': base_heads,
            'num_masked_heads': hard_num_masked_heads,
            'actual_params': verification['transformer']['actual_params'],
        },
        'mamba': {
            'model': 'mamba',
            'hidden_size': matched['mamba']['hidden_size'],
            'layers': matched['mamba']['layers'],
            'heads': max(4, matched['mamba']['hidden_size'] // 64),
            'state_dim': matched['mamba']['state_dim'],
            'actual_params': verification['mamba']['actual_params'],
        },
    }
    return final
