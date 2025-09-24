import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('default')
sns.set_palette("husl")

# Color scheme for different models
MODEL_COLORS = {
    'transformer_rope': '#1f77b4',
    'transformer_nope': '#ff7f0e',
    'transformer_alibi': '#2ca02c',
    'transformer_hard_alibi': '#d62728',
    'mamba': '#9467bd',
}

MODEL_LABELS = {
    'transformer_rope': 'Transformer: RoPE',
    'transformer_nope': 'Transformer: No PE',
    'transformer_alibi': 'Transformer: ALiBi',
    'transformer_hard_alibi': 'Transformer: Hard ALiBi',
    'mamba': 'GSSM: Mamba',
}


def plot_paper_reproduction(transformer_results, mamba_results, save_path="paper_reproduction.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    transformer_color = '#1f77b4'
    mamba_color = '#d62728'

    t_acc_examples = transformer_results['training'].get('accuracy_training_examples', [])
    t_accuracies = transformer_results['training'].get('accuracies', [])
    m_acc_examples = mamba_results['training'].get('accuracy_training_examples', [])
    m_accuracies = mamba_results['training'].get('accuracies', [])

    if t_acc_examples and t_accuracies:
        ax1.plot(t_acc_examples, t_accuracies, color=transformer_color, linewidth=2.5,
                 marker='o', markersize=4, label='Transformer: RoPE')
    if m_acc_examples and m_accuracies:
        ax1.plot(m_acc_examples, m_accuracies, color=mamba_color, linewidth=2.5,
                 marker='s', markersize=4, label='GSSM: Mamba')

    ax1.set_xlabel('Number of training examples')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('(a) Copying: training efficiency.')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    key_points = [1e3, 1e4, 1e5]
    max_examples = 0
    if t_acc_examples:
        max_examples = max(max_examples, max(t_acc_examples))
    if m_acc_examples:
        max_examples = max(max_examples, max(m_acc_examples))
    for point in key_points:
        if point <= max_examples:
            ax1.axvline(x=point, color='gray', alpha=0.3, linestyle='--')

    t_lengths = [r['length'] for r in transformer_results['length_gen']]
    t_accs = [r['accuracy'] for r in transformer_results['length_gen']]
    m_lengths = [r['length'] for r in mamba_results['length_gen']]
    m_accs = [r['accuracy'] for r in mamba_results['length_gen']]

    ax2.plot(t_lengths, t_accs, color=transformer_color, linewidth=2.5,
             marker='o', markersize=4, label='Transformer: RoPE')
    ax2.plot(m_lengths, m_accs, color=mamba_color, linewidth=2.5,
             marker='s', markersize=4, label='GSSM: Mamba')

    ax2.set_xlabel('Number of characters in string')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('(b) Copying: length generalization')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    train_max = 50
    ax2.axvline(x=train_max, color='gray', alpha=0.5, linestyle='--',
                label=f'Max train length ({train_max})')

    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=10)
        ax.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_comparison_table(transformer_results, mamba_results):
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON TABLE")
    print("=" * 80)

    print("\nMODEL SPECIFICATIONS:")
    print("-" * 50)
    print(f"{'Metric':<25} {'Transformer':<15} {'Mamba':<15} {'Ratio':<10}")
    print("-" * 50)

    t_params = transformer_results['param_count']
    m_params = mamba_results['param_count']
    print(f"{'Parameters':<25} {t_params:<15,} {m_params:<15,} {m_params/t_params:<10.3f}")

    print("\nTRAINING RESULTS:")
    print("-" * 50)
    t_time = transformer_results['training']['training_time']
    m_time = mamba_results['training']['training_time']
    print(f"{'Training Time (s)':<25} {t_time:<15.1f} {m_time:<15.1f} {m_time/t_time:<10.3f}")

    t_final_loss = transformer_results['training']['losses'][-1]
    m_final_loss = mamba_results['training']['losses'][-1]
    print(f"{'Final Loss':<25} {t_final_loss:<15.4f} {m_final_loss:<15.4f} {m_final_loss/t_final_loss:<10.3f}")

    print("\nLENGTH GENERALIZATION:")
    print("-" * 50)
    t_accs = [r['accuracy'] for r in transformer_results['length_gen']]
    m_accs = [r['accuracy'] for r in mamba_results['length_gen']]
    t_best = max(t_accs)
    m_best = max(m_accs)
    print(f"{'Best Accuracy (%)':<25} {t_best:<15.1f} {m_best:<15.1f} {m_best/t_best:<10.3f}")
    t_avg = np.mean(t_accs)
    m_avg = np.mean(m_accs)
    print(f"{'Average Accuracy (%)':<25} {t_avg:<15.1f} {m_avg:<15.1f} {m_avg/t_avg:<10.3f}")


def plot_length_gen_single(results, label, save_path="length_generalization.png"):
    """Plot length generalization for a single model."""
    lengths = [r['length'] for r in results['length_gen']]
    accs = [r['accuracy'] for r in results['length_gen']]

    plt.figure(figsize=(6, 4))
    plt.plot(lengths, accs, color='#1f77b4', linewidth=2.5, marker='o', markersize=4, label=label)
    plt.xlabel('Number of characters in string')
    plt.ylabel('Accuracy (%)')
    plt.title('Copying: length generalization')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def _extract_length_stats(results):
    try:
        cfg = results.get('config', {})
        train_len = int(cfg.get('max_train_len', 50))
    except Exception:
        train_len = 50

    lg = results.get('length_gen', []) or []
    lengths = [r['length'] for r in lg]
    accs = [r['accuracy'] for r in lg]
    best = max(accs) if accs else 0.0

    # Averages in three bands
    in_band = [a for l, a in zip(lengths, accs) if l <= train_len]
    near_band = [a for l, a in zip(lengths, accs) if train_len < l <= train_len + 10]
    far_band = [a for l, a in zip(lengths, accs) if l >= train_len + 30]

    def avg(x):
        return sum(x) / len(x) if x else 0.0

    # First length where accuracy < 50%
    drop_len = None
    for l, a in sorted(zip(lengths, accs)):
        if a < 50.0:
            drop_len = l
            break

    return {
        'train_len': train_len,
        'best': best,
        'avg_in': avg(in_band),
        'avg_near': avg(near_band),
        'avg_far': avg(far_band),
        'drop_len': drop_len,
        'lengths': lengths,
        'accs': accs,
    }


def plot_comprehensive_summary(transformer_results, mamba_results, save_path="comprehensive_summary.png"):
    """Top row: training loss, training time, model size. Bottom: detailed summary table."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.2], hspace=0.35, wspace=0.35)

    # Colors
    transformer_color = '#1f77b4'
    mamba_color = '#d62728'

    # ---------- Panel 1: Training Loss ----------
    ax1 = fig.add_subplot(gs[0, 0])
    if transformer_results is not None:
        t_ex = transformer_results['training'].get('training_examples', [])
        t_ls = transformer_results['training'].get('losses', [])
        ax1.plot(t_ex, t_ls, color=transformer_color, linewidth=2, label='Transformer')
    if mamba_results is not None:
        m_ex = mamba_results['training'].get('training_examples', [])
        m_ls = mamba_results['training'].get('losses', [])
        ax1.plot(m_ex, m_ls, color=mamba_color, linewidth=2, label='Mamba')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Training Examples')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ---------- Panel 2: Training Time ----------
    ax2 = fig.add_subplot(gs[0, 1])
    models = []
    times = []
    colors = []
    if transformer_results is not None:
        models.append('Transformer')
        times.append(transformer_results['training']['training_time'])
        colors.append(transformer_color)
    if mamba_results is not None:
        models.append('Mamba')
        times.append(mamba_results['training']['training_time'])
        colors.append(mamba_color)
    bars = ax2.bar(models, times, color=colors, alpha=0.8)
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Training Time Comparison')
    for bar, t in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{t:.1f}s', ha='center', va='bottom')
    ax2.grid(True, axis='y', alpha=0.2)

    # ---------- Panel 3: Model Size ----------
    ax3 = fig.add_subplot(gs[0, 2])
    params = []
    if transformer_results is not None:
        params.append(transformer_results.get('param_count', 0))
    if mamba_results is not None:
        params.append(mamba_results.get('param_count', 0))
    bars3 = ax3.bar(models, params, color=colors, alpha=0.8)
    ax3.set_ylabel('Parameters')
    ax3.set_title('Model Size Comparison')
    for bar, p in zip(bars3, params):
        ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{p:,}', ha='center', va='bottom')
    ax3.grid(True, axis='y', alpha=0.2)

    # ---------- Bottom: Summary Table ----------
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')

    # Extract summary stats
    lines = []
    lines.append('EXPERIMENT SUMMARY')
    lines.append('')
    lines.append('Model Configurations:')
    if transformer_results is not None:
        t_cfg = transformer_results.get('config', {})
        lines.append(f"• Transformer: {transformer_results.get('param_count', 0):,} params | H={t_cfg.get('hidden_size','?')}, L={t_cfg.get('layers','?')}, heads={t_cfg.get('heads','?')}")
    if mamba_results is not None:
        m_cfg = mamba_results.get('config', {})
        lines.append(f"• Mamba: {mamba_results.get('param_count', 0):,} params | H={m_cfg.get('hidden_size','?')}, L={m_cfg.get('layers','?')}, d_state={m_cfg.get('state_dim','?')}")
    lines.append('')
    lines.append('Training Results:')
    if transformer_results is not None:
        lines.append(f"• Transformer: {transformer_results['training']['losses'][-1]:.4f} final loss; time {transformer_results['training']['training_time']:.1f}s")
    if mamba_results is not None:
        lines.append(f"• Mamba: {mamba_results['training']['losses'][-1]:.4f} final loss; time {mamba_results['training']['training_time']:.1f}s")
    lines.append('')
    if transformer_results is not None:
        t_stats = _extract_length_stats(transformer_results)
        t_cfg = transformer_results.get('config', {})
        lines.append('Length Generalization (Transformer):')
        lines.append(f"• Best: {t_stats['best']:.1f}% | Avg ≤{t_stats['train_len']}: {t_stats['avg_in']:.1f}% | Near (+10): {t_stats['avg_near']:.1f}% | Far (≥{t_stats['train_len']+30}): {t_stats['avg_far']:.1f}%")
        lines.append(f"• <50% accuracy at length: {t_stats['drop_len']}")
        # Snapshot at key lengths
        def nearest(lengths, accs, target):
            if not lengths:
                return None, None
            idx = int(np.argmin(np.abs(np.array(lengths) - target)))
            return lengths[idx], accs[idx]
        snap_targets = [t_stats['train_len'], t_stats['train_len'] + 10, t_stats['train_len'] + 30]
        snaps = []
        for L in snap_targets:
            Lnear, Aval = nearest(t_stats['lengths'], t_stats['accs'], L)
            if Lnear is None:
                continue
            label = f"L={L}" if Lnear == L else f"L≈{Lnear}"
            snaps.append(f"{label}: {Aval:.1f}%")
        if snaps:
            lines.append("• Snapshot: " + ";  ".join(snaps))
        # Training config details
        lines.append('Training Config (Transformer):')
        lines.append(f"• Steps: {t_cfg.get('steps','?')} | LR: {t_cfg.get('lr','?')} | Train BS: {t_cfg.get('train_batch_size','?')} | Eval BS: {t_cfg.get('eval_batch_size','?')}")
        lines.append(f"• Context: {t_cfg.get('context_len','?')} | Eval Context: {t_cfg.get('eval_context_len','?')} | Train Len: {t_cfg.get('min_train_len','?')}-{t_cfg.get('max_train_len','?')} | Eval Len: {t_cfg.get('min_eval_len','?')}-{t_cfg.get('max_eval_len','?')}")
    if mamba_results is not None:
        m_stats = _extract_length_stats(mamba_results)
        m_cfg = mamba_results.get('config', {})
        lines.append('Length Generalization (Mamba):')
        lines.append(f"• Best: {m_stats['best']:.1f}% | Avg ≤{m_stats['train_len']}: {m_stats['avg_in']:.1f}% | Near (+10): {m_stats['avg_near']:.1f}% | Far (≥{m_stats['train_len']+30}): {m_stats['avg_far']:.1f}%")
        lines.append(f"• <50% accuracy at length: {m_stats['drop_len']}")
        # Snapshot at key lengths
        def nearest2(lengths, accs, target):
            if not lengths:
                return None, None
            idx = int(np.argmin(np.abs(np.array(lengths) - target)))
            return lengths[idx], accs[idx]
        snap_targets_m = [m_stats['train_len'], m_stats['train_len'] + 10, m_stats['train_len'] + 30]
        snaps_m = []
        for L in snap_targets_m:
            Lnear, Aval = nearest2(m_stats['lengths'], m_stats['accs'], L)
            if Lnear is None:
                continue
            label = f"L={L}" if Lnear == L else f"L≈{Lnear}"
            snaps_m.append(f"{label}: {Aval:.1f}%")
        if snaps_m:
            lines.append("• Snapshot: " + ";  ".join(snaps_m))
        # Training config details
        lines.append('Training Config (Mamba):')
        lines.append(f"• Steps: {m_cfg.get('steps','?')} | LR: {m_cfg.get('lr','?')} | Train BS: {m_cfg.get('train_batch_size','?')} | Eval BS: {m_cfg.get('eval_batch_size','?')}")
        lines.append(f"• Context: {m_cfg.get('context_len','?')} | Eval Context: {m_cfg.get('eval_context_len','?')} | Train Len: {m_cfg.get('min_train_len','?')}-{m_cfg.get('max_train_len','?')} | Eval Len: {m_cfg.get('min_eval_len','?')}-{m_cfg.get('max_eval_len','?')}")

    # Render text block
    text = "\n".join(lines)
    ax4.text(0.02, 0.98, text, va='top', ha='left', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.35))

    fig.suptitle('Comprehensive Model Comparison (Top: 3 panels; Bottom: Summary Table)', fontsize=16, y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_comparison(transformer_results, mamba_results, save_path="comparison_overview.png"):
    """Side-by-side comparison: loss, training efficiency, and length generalization."""
    if transformer_results is None or mamba_results is None:
        # Nothing to compare
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    (ax_loss, ax_eff), (ax_len, ax_misc) = axes

    transformer_color = '#1f77b4'
    mamba_color = '#d62728'

    # Loss overlay
    t_ex = transformer_results['training'].get('training_examples', [])
    t_ls = transformer_results['training'].get('losses', [])
    m_ex = mamba_results['training'].get('training_examples', [])
    m_ls = mamba_results['training'].get('losses', [])
    ax_loss.plot(t_ex, t_ls, color=transformer_color, linewidth=2, label='Transformer')
    ax_loss.plot(m_ex, m_ls, color=mamba_color, linewidth=2, label='Mamba')
    ax_loss.set_title('Training Loss')
    ax_loss.set_xlabel('Training Examples')
    ax_loss.set_ylabel('Loss')
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    # Training efficiency (checkpoint accuracies vs examples)
    t_ax = transformer_results['training'].get('accuracy_training_examples', [])
    t_acc = transformer_results['training'].get('accuracies', [])
    m_ax = mamba_results['training'].get('accuracy_training_examples', [])
    m_acc = mamba_results['training'].get('accuracies', [])
    if t_ax and t_acc:
        ax_eff.plot(t_ax, t_acc, color=transformer_color, linewidth=2, marker='o', label='Transformer')
    if m_ax and m_acc:
        ax_eff.plot(m_ax, m_acc, color=mamba_color, linewidth=2, marker='s', label='Mamba')
    ax_eff.set_title('Training Efficiency (Accuracy vs Examples)')
    ax_eff.set_xlabel('Training Examples')
    ax_eff.set_ylabel('Accuracy (%)')
    ax_eff.set_ylim(0, 100)
    ax_eff.grid(True, alpha=0.3)
    ax_eff.legend()

    # Length generalization overlay
    t_lengths = [r['length'] for r in transformer_results['length_gen']]
    t_accs = [r['accuracy'] for r in transformer_results['length_gen']]
    m_lengths = [r['length'] for r in mamba_results['length_gen']]
    m_accs = [r['accuracy'] for r in mamba_results['length_gen']]
    ax_len.plot(t_lengths, t_accs, color=transformer_color, linewidth=2, marker='o', label='Transformer')
    ax_len.plot(m_lengths, m_accs, color=mamba_color, linewidth=2, marker='s', label='Mamba')
    ax_len.set_title('Length Generalization')
    ax_len.set_xlabel('Sequence Length')
    ax_len.set_ylabel('Accuracy (%)')
    ax_len.set_ylim(0, 105)
    ax_len.grid(True, alpha=0.3)
    ax_len.legend()

    # Misc panel: parameter compare + times
    models = ['Transformer', 'Mamba']
    params = [transformer_results.get('param_count', 0), mamba_results.get('param_count', 0)]
    times = [transformer_results['training']['training_time'], mamba_results['training']['training_time']]
    x = np.arange(len(models))
    width = 0.35
    ax_misc.bar(x - width/2, params, width, label='Params', color=['#8fbce6', '#f0a7a7'])
    ax_misc2 = ax_misc.twinx()
    ax_misc2.bar(x + width/2, times, width, label='Time (s)', color=['#2c7bb6', '#d7191c'])
    ax_misc.set_xticks(x)
    ax_misc.set_xticklabels(models)
    ax_misc.set_ylabel('Parameters')
    ax_misc2.set_ylabel('Training Time (s)')
    ax_misc.set_title('Model Size and Time')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_all_models_comparison(results_dict, save_path="all_models_comparison.png"):
    """Create comprehensive comparison plot for all models"""

    # Filter out None results
    valid_results = {k: v for k, v in results_dict.items() if v is not None}

    if len(valid_results) < 1:
        print("No valid results to plot")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Training Loss Curves
    for model_name, results in valid_results.items():
        if 'training' in results and 'losses' in results['training']:
            losses = results['training']['losses']
            examples = results['training']['training_examples']
            color = MODEL_COLORS.get(model_name, 'gray')
            label = MODEL_LABELS.get(model_name, model_name)
            ax1.plot(examples, losses, color=color, linewidth=2, label=label, alpha=0.8)

    ax1.set_xlabel('Training Examples')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Training Accuracy (if available)
    for model_name, results in valid_results.items():
        if ('training' in results and 'accuracies' in results['training'] and
            'accuracy_training_examples' in results['training']):
            accuracies = results['training']['accuracies']
            examples = results['training']['accuracy_training_examples']
            if accuracies and examples:
                color = MODEL_COLORS.get(model_name, 'gray')
                label = MODEL_LABELS.get(model_name, model_name)
                ax2.plot(examples, accuracies, color=color, linewidth=2,
                        marker='o', markersize=4, label=label, alpha=0.8)

    ax2.set_xlabel('Training Examples')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy Progress')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    # Plot 3: Length Generalization
    for model_name, results in valid_results.items():
        if 'length_gen' in results and results['length_gen']:
            lengths = [r['length'] for r in results['length_gen']]
            accuracies = [r['accuracy'] for r in results['length_gen']]
            color = MODEL_COLORS.get(model_name, 'gray')
            label = MODEL_LABELS.get(model_name, model_name)
            ax3.plot(lengths, accuracies, color=color, linewidth=2,
                    marker='s', markersize=3, label=label, alpha=0.8)

    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Length Generalization')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)

    # Plot 4: Final Performance Summary
    model_names = []
    final_accuracies = []
    param_counts = []

    for model_name, results in valid_results.items():
        if 'fixed_accuracy' in results:
            model_names.append(MODEL_LABELS.get(model_name, model_name))
            final_accuracies.append(results['fixed_accuracy'])
            param_counts.append(results.get('param_count', 0))

    if model_names:
        bars = ax4.bar(range(len(model_names)), final_accuracies,
                      color=[MODEL_COLORS.get(k, 'gray') for k in valid_results.keys()],
                      alpha=0.7)
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Final Accuracy (%)')
        ax4.set_title('Final Performance Comparison')
        ax4.set_xticks(range(len(model_names)))
        ax4.set_xticklabels(model_names, rotation=45, ha='right')
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3, axis='y')

        # Add parameter counts as text on bars
        for i, (bar, params) in enumerate(zip(bars, param_counts)):
            if params > 0:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{params/1e6:.1f}M', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"All models comparison plot saved to: {save_path}")
