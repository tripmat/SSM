#!/usr/bin/env python3
"""
Reproduce all figures from existing checkpoint data without retraining models.

This script demonstrates that once you have the checkpoint JSON files,
you can regenerate all plots and analyses without needing the model weights
or rerunning training.
"""

import json
import os
import sys
from pathlib import Path

# Add the project to path for imports
sys.path.append(str(Path(__file__).parent))

from ssm_experiments.utils.plotting import (
    plot_paper_reproduction,
    plot_comprehensive_summary,
    plot_comparison,
    plot_length_gen_single,
    plot_all_models_comparison,
    create_comparison_table
)


def load_checkpoint_results(checkpoint_dir="outputs/checkpoints"):
    """Load all available checkpoint results"""
    results = {}

    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return results

    for filename in os.listdir(checkpoint_dir):
        if not filename.endswith('_results.json'):
            continue

        # Parse model name from filename
        parts = filename.replace('_results.json', '').split('_')
        if parts[0] == 'transformer':
            # Handle transformer variants: transformer_rope, transformer_nope, etc.
            if len(parts) > 1 and parts[1] in ['rope', 'nope', 'alibi', 'hard']:
                if parts[1] == 'hard' and len(parts) > 2 and parts[2] == 'alibi':
                    model_name = 'transformer_hard_alibi'
                else:
                    model_name = f'transformer_{parts[1]}'
            else:
                model_name = 'transformer_rope'  # default
        else:
            model_name = parts[0]  # e.g., mamba, paper_mamba

        filepath = os.path.join(checkpoint_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Only keep the most complete results (highest step count)
            if model_name in results:
                current_steps = len(results[model_name]['training']['losses'])
                new_steps = len(data['training']['losses'])
                if new_steps <= current_steps:
                    continue  # Skip if not better

            results[model_name] = data
            print(f"Loaded {model_name}: {len(data['training']['losses'])} training steps")

        except Exception as e:
            print(f"Failed to load {filename}: {e}")

    return results


def reproduce_all_figures(results, output_dir="reproduced_figures"):
    """Reproduce all possible figures from checkpoint data"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating figures in {output_dir}/")

    figures_generated = []

    # 1. All-models comparison (requires 2+ models)
    if len(results) >= 2:
        try:
            path = os.path.join(output_dir, "all_models_comparison.png")
            plot_all_models_comparison(results, path)
            figures_generated.append("all_models_comparison.png")
            print(f"Generated: all_models_comparison.png")
        except Exception as e:
            print(f"Failed to generate all_models_comparison: {e}")

    # 2. Traditional transformer vs mamba plots (for backward compatibility)
    if 'transformer_rope' in results and 'mamba' in results:
        t_results = results['transformer_rope']
        m_results = results['mamba']

        try:
            path = os.path.join(output_dir, "paper_reproduction.png")
            plot_paper_reproduction(t_results, m_results, path)
            figures_generated.append("paper_reproduction.png")
            print(f"OK: Generated: paper_reproduction.png")
        except Exception as e:
            print(f"ERROR: Failed to generate paper_reproduction: {e}")

        try:
            path = os.path.join(output_dir, "comprehensive_summary.png")
            plot_comprehensive_summary(t_results, m_results, path)
            figures_generated.append("comprehensive_summary.png")
            print(f"OK: Generated: comprehensive_summary.png")
        except Exception as e:
            print(f"ERROR: Failed to generate comprehensive_summary: {e}")

        try:
            path = os.path.join(output_dir, "comparison_overview.png")
            plot_comparison(t_results, m_results, path)
            figures_generated.append("comparison_overview.png")
            print(f"OK: Generated: comparison_overview.png")
        except Exception as e:
            print(f"ERROR: Failed to generate comparison_overview: {e}")

        # Text-based comparison table
        try:
            print("\n" + "="*60)
            print("TRANSFORMER vs MAMBA COMPARISON TABLE")
            print("="*60)
            create_comparison_table(t_results, m_results)
            print("OK: Generated: comparison table (printed above)")
        except Exception as e:
            print(f"ERROR: Failed to generate comparison table: {e}")

    # 3. Single model length generalization plots
    for model_name, model_results in results.items():
        if 'length_gen' in model_results and model_results['length_gen']:
            try:
                path = os.path.join(output_dir, f"{model_name}_length_generalization.png")
                label = model_name.replace('_', ' ').title()
                plot_length_gen_single(model_results, label, path)
                figures_generated.append(f"{model_name}_length_generalization.png")
                print(f"OK: Generated: {model_name}_length_generalization.png")
            except Exception as e:
                print(f"ERROR: Failed to generate {model_name} length plot: {e}")

    # 4. Single model comprehensive summaries
    for model_name, model_results in results.items():
        try:
            path = os.path.join(output_dir, f"{model_name}_summary.png")
            plot_comprehensive_summary(model_results, None, path)
            figures_generated.append(f"{model_name}_summary.png")
            print(f"OK: Generated: {model_name}_summary.png")
        except Exception as e:
            print(f"ERROR: Failed to generate {model_name} summary: {e}")

    return figures_generated


def main():
    print("Reproducing all figures from existing checkpoint data...")
    print("="*70)

    # Load all available results
    results = load_checkpoint_results()

    if not results:
        print("\nNo checkpoint results found!")
        print("Make sure you have run experiments and have *_results.json files in outputs/checkpoints/")
        return

    print(f"\nSummary: Loaded {len(results)} models with complete training data")
    for name, data in results.items():
        steps = len(data['training']['losses'])
        time = data['training']['training_time']
        params = data['param_count']
        print(f"  - {name}: {steps} steps, {time:.1f}s training, {params:,} params")

    # Generate all figures
    figures = reproduce_all_figures(results)

    print(f"\nSUCCESS! Generated {len(figures)} figures from checkpoint data:")
    for fig in figures:
        print(f"  -> reproduced_figures/{fig}")

    print(f"\nKey insight: All figures were reproduced WITHOUT retraining any models!")
    print(f"The checkpoint JSON files contain ALL necessary data for visualization.")


if __name__ == "__main__":
    main()
