# SSM Experiments (Slim)

A clean, CPU-friendly reproduction of the Transformer vs Mamba comparison on a synthetic copy task, based on the "Repeat After Me" study by Jelassi et al. This is a slim, stand-alone layout distilled from exploratory iterations, keeping only what's needed to run, train, evaluate, and plot.

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
    - `paper_mamba.py` – CPU-friendly, paper-informed Mamba approximation
  - `utils/` – plotting and parameter matching
- `outputs/` – generated artifacts (gitignored)

## Notes
- Mamba path: This repo uses a paper-informed Mamba implementation (no external `mamba_ssm` dependency). The `mamba` option aliases to the same implementation as `paper_mamba` for consistency and Windows compatibility.
- Mamba configuration via JSON: you can override per-model parameters in `--config` (see `configs/experiment.example.json`). For Mamba variants, you can set `state_dim`, `dt_min`, and `dt_max`.
- Reproducibility: Configs and parameter counts are saved with results.
- This slim package intentionally omits unused legacy files and large artifacts. The original exploratory files remain in the repo root but are not needed to run via `ssm_experiments`.

## Citation & Original Work

This repository is a slim reproduction of experiments from:

**"Repeat After Me: Transformers are Better than State Space Models at Copying"**
*Samy Jelassi, David Brandfonbrener, Sham M. Kakade, Eran Malach*

- **Paper**: [arXiv:2402.01032](https://arxiv.org/abs/2402.01032) (2024)
- **Original Repository**: [jelassi/repeat-after-me](https://github.com/jelassi/repeat-after-me)

### Key Differences from Original
- **CPU-friendly implementation**: No CUDA-specific dependencies required
- **Simplified architecture**: Focused on core comparison without auxiliary experiments
- **Progressive plotting**: Real-time visualization as models complete training
- **Enhanced reproducibility**: Complete parameter matching and checkpoint management
- **Streamlined workflow**: Single command execution with flexible model selection

If you use this reproduction in your research, please cite the original paper:

```bibtex
@article{jelassi2024repeat,
  title={Repeat After Me: Transformers are Better than State Space Models at Copying},
  author={Jelassi, Samy and Brandfonbrener, David and Kakade, Sham M. and Malach, Eran},
  journal={arXiv preprint arXiv:2402.01032},
  year={2024}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
