"""
Single-source Python configuration for SSM experiments.

This file is loaded automatically if you do not pass --config.
Edit the values below to control global and per-model parameters.

Note: This is a Python config (not JSON), so we can derive
context sizes from the chosen max lengths. We keep the final
CONFIG as plain values so it stays JSON-serializable.
"""

# ----- Derived length settings (Solution 1: variables, then CONFIG) -----

MIN_TRAIN_LEN = 5
MAX_TRAIN_LEN = 20

MIN_EVAL_LEN = 3
MAX_EVAL_LEN = 41

# Ensure context is strictly greater than 2x the max length, plus cushion
CONTEXT_LEN = 2 * MAX_TRAIN_LEN + 20      # > 2 * MAX_TRAIN_LEN
EVAL_CONTEXT_LEN = 2 * MAX_EVAL_LEN + 50  # > 2 * MAX_EVAL_LEN

CONFIG = {
    "global": {
        "target_params": 350_000,
        "steps": 20_000,
        "max_lr": 1e-4,
        "min_lr": 1e-6,
        "warmup_steps": 300,
        "train_batch_size": 32,
        "eval_batch_size": 4,
        "eval_num_batches": 3,
        "min_train_len": MIN_TRAIN_LEN ,
        "max_train_len": MAX_TRAIN_LEN,
        "min_eval_len": MIN_EVAL_LEN ,
        "max_eval_len": MAX_EVAL_LEN,
        "context_len": CONTEXT_LEN,
        "eval_context_len": EVAL_CONTEXT_LEN,
        "vocab_size": 26,
        "n_gram": 0,
        "length_answer": 0,
    },
    "models": {
        "transformer_rope": {"heads": 4},
        "transformer_nope": {"heads": 4},
        "transformer_alibi": {"heads": 4},
        "transformer_hard_alibi": {"heads": 4, "num_masked_heads": 2},
        "paper_mamba": {"state_dim": 16, "dt_min": 0.001, "dt_max": 0.1},
        "minimal_mamba": {"state_dim": 16, "dt_min": 0.001, "dt_max": 0.1, "expand": 2, "d_conv": 4},
    },
}
