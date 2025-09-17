# SSM Experiments (Slim)

A clean, CPU-friendly reproduction of the Transformer vs Mamba comparison on a synthetic copy task. This is a slim, stand-alone layout distilled from exploratory iterations, keeping only what’s needed to run, train, evaluate, and plot.

## Quickstart

- Install deps:
  - `pip install -r requirements.txt`
- Run the experiment (trains Transformer RoPE and Mamba, saves outputs under `outputs/`):
  - `python -m ssm_experiments.cli`

Optional flags:
- `--steps 2000` total training steps
- `--lr 0.000125` learning rate (default)
- `--train-batch-size 4` training batch size
- `--eval-batch-size 4` evaluation batch size

Example:
- `python -m ssm_experiments.cli --steps 2000 --train-batch-size 4`

## What It Does
- Trains matched-parameter Transformer (RoPE) and Mamba models on a synthetic copy task.
- Saves checkpoints and JSON results to `outputs/checkpoints/`.
- Evaluates length generalization and fixed-set accuracy.
- Generates paper-style plots to `outputs/figures/`.

## Repo Layout (Slim)
- `ssm_experiments/`
  - `cli.py` – command-line entry and experiment orchestration
  - `train.py` – CPU training loop with checkpointing
  - `evaluate.py` – accuracy and length-generalization evaluation
  - `data/` – tokenizer and synthetic datasets
  - `models/` – model registry and minimal implementations
    - `lstm.py` – simple LSTM baseline (optional)
    - `paper_mamba.py` – CPU-friendly, paper-informed Mamba approximation
  - `utils/` – plotting and parameter matching
- `outputs/` – generated artifacts (gitignored)

## Notes
- Real Mamba: If `mamba_ssm` is installed, it is used automatically. Otherwise a CPU-friendly paper-informed approximation is used.
- Reproducibility: Configs and parameter counts are saved with results.
- This slim package intentionally omits unused legacy files and large artifacts. The original exploratory files remain in the repo root but are not needed to run via `ssm_experiments`.

## License
See LICENSE in the original project if you bring in licensed components. This slim package itself has no license header; add one if needed for distribution.
