"""Render README latency cards: uv run --with matplotlib scripts/render-model-benchmarks.py."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
ROWS = json.loads((ROOT / 'assets/model-benchmarks.json').read_text())
THEMES = {
    'dark': dict(bg='#0d1117', fg='#eeeeec', muted='#aaa9a1', track='#25292e', ref='#70716c', border='#2c3138'),
    'light': dict(bg='#ffffff', fg='#20252b', muted='#656962', track='#eef0f2', ref='#989b96', border='#e2e5e8'),
}
BLUE = '#3589e4'
for theme, c in THEMES.items():
    fig = plt.figure(figsize=(18, 14.6), facecolor=c['bg'])
    fig.text(.025, .963, 'Models on Vulkan', fontsize=27, weight='bold', color=c['fg'])
    fig.text(.975, .967, 'RTX 4070  ·  inference latency  ·  lower is better', ha='right', fontsize=12, color=c['muted'])
    fig.text(.025, .933, 'BEFORE', fontsize=10, weight='bold', color=c['ref'])
    fig.text(.084, .933, 'reference backend', fontsize=10, color=c['muted'])
    fig.text(.225, .933, 'AFTER', fontsize=10, weight='bold', color=BLUE)
    fig.text(.273, .933, 'onnx-vulkan-rs', fontsize=10, color=c['muted'])
    for i, row in enumerate(ROWS):
        col, r = i % 3, i // 3
        ax = fig.add_axes([.025 + col * .325, .744 - r * .169, .3, .154])
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis('off')
        ax.text(0, .96, row['model'], va='top', fontsize=18, weight='bold', color=c['fg'])
        ax.text(0, .80, row['profile'], va='top', fontsize=10.5, color=c['muted'])
        before, after = row['before_ms'], row['after_ms']
        gain = (1 - after / before) * 100
        label = f'{gain:.0f}% less time' if gain >= 0 else f'{-gain:.0f}% more time'
        if row['before_label'].startswith('~'):
            label = '~' + label
        ax.text(1, .79, label, ha='right', va='top', fontsize=10.5, weight='bold', color=BLUE if gain >= 0 else c['fg'])
        scale = max(before, after)
        for y, value, color in ((.53, before, c['ref']), (.25, after, BLUE)):
            ax.add_patch(Rectangle((0, y), 1, .09, color=c['track'], linewidth=0))
            ax.add_patch(Rectangle((0, y), value / scale, .09, color=color, linewidth=0))
        ax.text(0, .48, row['reference'], va='top', fontsize=11, color=c['muted'])
        ax.text(1, .48, f"{row['before_label']} ms", va='top', ha='right', fontsize=12, weight='bold', color=c['fg'])
        ax.text(0, .20, row['project'], va='top', fontsize=11, color=c['muted'])
        ax.text(1, .20, f"{row['after_label']} ms", va='top', ha='right', fontsize=12, weight='bold', color=BLUE)
        ax.plot([0, 1], [.015, .015], color=c['border'], lw=.7)
    fig.text(.025, .029, 'Each pair has its own linear scale. SAM 3: CUDA bf16 reference; all others: CPU reference. Piper: sentence 1, experimental INT8.', fontsize=10, color=c['muted'])
    fig.text(.025, .012, 'Historical runs · different workloads and protocols · quality checks and full data: BENCHMARKS.md', fontsize=10, color=c['muted'])
    fig.savefig(ROOT / f'assets/model-benchmarks-{theme}.png', dpi=160, facecolor=c['bg'])
    plt.close(fig)
