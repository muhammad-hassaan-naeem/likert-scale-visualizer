"""
likert_visualizer.py
─────────────────────────────────────────────────────────────────────────────
Likert Scale Survey Visualizer

Upload any Likert-scale CSV survey dataset and automatically generate
7 professional, publication-ready visualizations.

Author : Muhammad Hassaan Naeem
         ML Researcher | Lab Engineer @ SCAT
         github.com/muhammad-hassaan-naeem
         DOI: 10.1016/j.asej.2024.102794

Charts Generated:
  1.  Grouped Bar Chart        — mean scores per construct
  2.  Diverging Stacked Bar    — response distribution (strongly disagree → agree)
  3.  Radar / Spider Chart     — construct mean scores
  4.  Heatmap                  — item-level mean scores
  5.  Distribution Histogram   — response frequency for each construct
  6.  Box Plot                 — score spread per construct
  7.  Demographic Breakdown    — performance by group (if demographic cols exist)

Usage:
    python likert_visualizer.py --file sample_survey.csv --constructs config.json
    python likert_visualizer.py --file sample_survey.csv  (auto-detect mode)
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats

# ── Dark theme ────────────────────────────────────────────────────────────────
DARK = {
    'bg':      '#0d1117',
    'surface': '#161b22',
    'surface2':'#1f2937',
    'border':  '#30363d',
    'text':    '#f0f6fc',
    'muted':   '#8b949e',
    'dim':     '#484f58',
    'green':   '#2ea043',
    'blue':    '#388bfd',
    'red':     '#f85149',
    'yellow':  '#d29922',
    'purple':  '#bc8cff',
    'teal':    '#39d353',
    'orange':  '#e3b341',
}

# Colour palette for constructs
PALETTE = [
    '#2ea043', '#388bfd', '#bc8cff',
    '#d29922', '#f85149', '#39d353',
    '#e3b341', '#58a6ff', '#ff7c38',
]

# Diverging palette for stacked bars
DIV_PALETTE = ['#c0392b', '#e74c3c', '#95a5a6', '#27ae60', '#1e8449']

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         9,
    'axes.facecolor':    DARK['surface'],
    'figure.facecolor':  DARK['bg'],
    'axes.edgecolor':    DARK['border'],
    'axes.labelcolor':   DARK['muted'],
    'axes.titlecolor':   DARK['text'],
    'axes.titlesize':    11,
    'axes.titleweight':  'bold',
    'xtick.color':       DARK['muted'],
    'ytick.color':       DARK['muted'],
    'text.color':        DARK['muted'],
    'grid.color':        DARK['border'],
    'grid.alpha':        0.4,
    'grid.linestyle':    '--',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'legend.facecolor':  DARK['surface'],
    'legend.edgecolor':  DARK['border'],
    'legend.labelcolor': DARK['text'],
})


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def auto_detect_constructs(df: pd.DataFrame) -> dict:
    """
    Auto-detect Likert constructs from column names.
    Groups columns like SC1, SC2, SC3 → {'SC': ['SC1','SC2','SC3']}
    """
    constructs = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Try to find columns ending in digits (e.g. SC1, SC2, TC1...)
    import re
    groups = {}
    for col in numeric_cols:
        match = re.match(r'^([A-Za-z_]+)(\d+)$', col)
        if match:
            prefix = match.group(1).rstrip('_')
            groups.setdefault(prefix, []).append(col)

    if groups:
        # Only keep groups with 2+ items
        constructs = {k: v for k, v in groups.items() if len(v) >= 2}

    # Fallback: treat all numeric columns as one construct
    if not constructs:
        constructs = {'Survey': numeric_cols}

    return constructs


def compute_scores(df: pd.DataFrame, constructs: dict) -> pd.DataFrame:
    """Compute composite mean score per construct."""
    return pd.DataFrame({
        name: df[items].mean(axis=1)
        for name, items in constructs.items()
        if all(c in df.columns for c in items)
    })


def response_counts(df: pd.DataFrame, constructs: dict,
                    scale: int = 5) -> pd.DataFrame:
    """
    Count percentage of each Likert response (1–scale)
    across all items in each construct.
    Returns DataFrame: constructs × scale points
    """
    rows = []
    for name, items in constructs.items():
        valid_items = [c for c in items if c in df.columns]
        if not valid_items:
            continue
        all_responses = df[valid_items].values.flatten()
        total = len(all_responses)
        counts = {i: np.sum(all_responses == i) / total * 100
                  for i in range(1, scale + 1)}
        counts['Construct'] = name
        rows.append(counts)
    result = pd.DataFrame(rows).set_index('Construct')
    return result


def save(fig, name: str, out_dir: str):
    """Save figure as PNG."""
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=DARK['bg'])
    print(f"  ✅ Saved: {name}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  CHART 1 — Grouped Bar Chart (Mean Scores)
# ─────────────────────────────────────────────────────────────────────────────

def plot_mean_scores(scores: pd.DataFrame, out_dir: str):
    """Grouped bar chart showing mean ± std for each construct."""
    means = scores.mean()
    stds  = scores.std()
    n     = len(means)
    x     = np.arange(n)
    colors = PALETTE[:n]

    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), 5),
                            facecolor=DARK['bg'])
    bars = ax.bar(x, means.values, yerr=stds.values,
                  color=colors, edgecolor=DARK['bg'],
                  linewidth=0.8, width=0.55,
                  error_kw=dict(ecolor=DARK['muted'], capsize=4,
                                elinewidth=1.2))

    # Value labels
    for bar, mean, std in zip(bars, means.values, stds.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                mean + std + 0.05,
                f'{mean:.2f}', ha='center', va='bottom',
                fontsize=9, color=DARK['text'], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(means.index, fontsize=10)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel('Mean Score (Likert 1–5)')
    ax.set_title('Construct Mean Scores  (error bars = ±1 SD)')
    ax.axhline(3.0, color=DARK['dim'], linewidth=1,
               linestyle=':', label='Neutral midpoint (3.0)')
    ax.legend(fontsize=8)
    ax.grid(axis='y')
    fig.tight_layout()
    save(fig, '01_mean_scores.png', out_dir)


# ─────────────────────────────────────────────────────────────────────────────
#  CHART 2 — Diverging Stacked Bar
# ─────────────────────────────────────────────────────────────────────────────

def plot_diverging_stacked(df: pd.DataFrame, constructs: dict,
                            out_dir: str):
    """
    Diverging stacked bar chart.
    Responses 1–2 go left (negative), 3 stays centred, 4–5 go right.
    """
    rc = response_counts(df, constructs)
    labels_left  = ['Strongly Disagree (1)', 'Disagree (2)']
    labels_mid   = ['Neutral (3)']
    labels_right = ['Agree (4)', 'Strongly Agree (5)']
    all_labels   = labels_left + labels_mid + labels_right

    constructs_list = rc.index.tolist()
    n = len(constructs_list)

    fig, ax = plt.subplots(figsize=(12, max(4, n * 0.7 + 1)),
                            facecolor=DARK['bg'])

    y_pos = np.arange(n)
    left_starts = -(rc[1].values + rc[2].values)

    cumulative_left  = np.zeros(n)
    cumulative_right = np.zeros(n)

    # Draw negative side (1, 2)
    for col, color, label in zip([1, 2],
                                   [DIV_PALETTE[0], DIV_PALETTE[1]],
                                   labels_left):
        vals = rc[col].values
        ax.barh(y_pos, -vals, left=left_starts + cumulative_left,
                color=color, edgecolor=DARK['bg'],
                linewidth=0.5, height=0.6, label=label)
        cumulative_left += vals

    # Draw neutral (3)
    ax.barh(y_pos, rc[3].values, left=0,
            color=DIV_PALETTE[2], edgecolor=DARK['bg'],
            linewidth=0.5, height=0.6, label=labels_mid[0])

    # Draw positive side (4, 5)
    neutral_half = rc[3].values / 2
    cumulative_right = neutral_half.copy()
    for col, color, label in zip([4, 5],
                                   [DIV_PALETTE[3], DIV_PALETTE[4]],
                                   labels_right):
        vals = rc[col].values
        ax.barh(y_pos, vals, left=cumulative_right,
                color=color, edgecolor=DARK['bg'],
                linewidth=0.5, height=0.6, label=label)
        cumulative_right += vals

    ax.axvline(0, color=DARK['text'], linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(constructs_list, fontsize=10)
    ax.set_xlabel('Percentage of Responses (%)')
    ax.set_title('Response Distribution — Diverging Stacked Bar')

    # Fix x-axis labels to show positive values on both sides
    x_ticks = ax.get_xticks()
    ax.set_xticklabels([f'{abs(int(x))}%' for x in x_ticks])

    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    save(fig, '02_diverging_stacked.png', out_dir)


# ─────────────────────────────────────────────────────────────────────────────
#  CHART 3 — Radar / Spider Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_radar(scores: pd.DataFrame, out_dir: str):
    """Radar chart of construct mean scores."""
    means  = scores.mean()
    labels = list(means.index)
    N      = len(labels)
    values = list(means.values) + [means.values[0]]  # close the loop
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7),
                            subplot_kw=dict(polar=True),
                            facecolor=DARK['bg'])
    ax.set_facecolor(DARK['surface'])

    # Fill
    ax.fill(angles, values, alpha=0.2, color=DARK['green'])
    ax.plot(angles, values, 'o-', linewidth=2.5,
            color=DARK['green'], markersize=7,
            markerfacecolor=DARK['green'],
            markeredgecolor=DARK['bg'])

    # Gridlines
    ax.set_ylim(1, 5)
    ax.set_yticks([2, 3, 4, 5])
    ax.set_yticklabels(['2', '3', '4', '5'],
                        size=8, color=DARK['muted'])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11, color=DARK['text'])
    ax.grid(color=DARK['border'], alpha=0.5)
    ax.spines['polar'].set_color(DARK['border'])

    # Annotate values
    for angle, val, label in zip(angles[:-1], means.values, labels):
        ax.annotate(f'{val:.2f}',
                    xy=(angle, val),
                    xytext=(angle, val + 0.28),
                    fontsize=9, color=DARK['text'],
                    ha='center', fontweight='bold')

    ax.set_title('Construct Mean Scores — Radar Chart\n(Likert 1–5)',
                  color=DARK['text'], fontsize=12, pad=25)
    fig.tight_layout()
    save(fig, '03_radar_chart.png', out_dir)


# ─────────────────────────────────────────────────────────────────────────────
#  CHART 4 — Item-Level Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame, constructs: dict, out_dir: str):
    """Heatmap of mean score for every individual item."""
    rows = []
    for construct, items in constructs.items():
        valid = [c for c in items if c in df.columns]
        for item in valid:
            rows.append({
                'Construct': construct,
                'Item':      item,
                'Mean':      round(df[item].mean(), 2),
            })

    if not rows:
        return

    heat_df = pd.DataFrame(rows)
    pivot   = heat_df.pivot(index='Construct', columns='Item', values='Mean')

    fig, ax = plt.subplots(
        figsize=(max(8, len(pivot.columns) * 0.8 + 2),
                 max(4, len(pivot.index) * 0.7 + 1)),
        facecolor=DARK['bg']
    )

    cmap = LinearSegmentedColormap.from_list(
        'likert', ['#c0392b', '#f39c12', '#27ae60'], N=256
    )
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt='.2f',
        cmap=cmap, vmin=1, vmax=5,
        linewidths=0.8, linecolor=DARK['bg'],
        annot_kws={'size': 9, 'color': DARK['bg'], 'fontweight': 'bold'},
        cbar_kws={'shrink': 0.8, 'label': 'Mean Score'}
    )
    ax.set_facecolor(DARK['surface'])
    ax.set_title('Item-Level Mean Scores Heatmap',
                  color=DARK['text'])
    ax.tick_params(colors=DARK['muted'], rotation=30)
    fig.tight_layout()
    save(fig, '04_item_heatmap.png', out_dir)


# ─────────────────────────────────────────────────────────────────────────────
#  CHART 5 — Response Distribution Histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_histograms(scores: pd.DataFrame, out_dir: str):
    """One histogram + KDE per construct, arranged in a grid."""
    n_cols   = min(3, len(scores.columns))
    n_rows   = int(np.ceil(len(scores.columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 5, n_rows * 3.5),
                              facecolor=DARK['bg'])
    axes = np.array(axes).flatten()

    for i, (col, color) in enumerate(zip(scores.columns,
                                          PALETTE[:len(scores.columns)])):
        ax   = axes[i]
        data = scores[col].dropna()
        ax.set_facecolor(DARK['surface'])

        ax.hist(data, bins=15, color=color, alpha=0.65,
                edgecolor=DARK['bg'], density=True)

        # KDE
        if data.std() > 0:
            kde_x = np.linspace(max(1, data.min()), min(5, data.max()), 200)
            kde   = stats.gaussian_kde(data)
            ax.plot(kde_x, kde(kde_x), color=DARK['text'],
                    linewidth=2)

        ax.axvline(data.mean(), color=DARK['yellow'],
                   linewidth=1.8, linestyle='--',
                   label=f'Mean={data.mean():.2f}')
        ax.axvline(data.median(), color=DARK['purple'],
                   linewidth=1.5, linestyle=':',
                   label=f'Median={data.median():.2f}')

        ax.set_title(col, color=DARK['text'], fontsize=10)
        ax.set_xlabel('Score (1–5)', color=DARK['muted'])
        ax.set_ylabel('Density', color=DARK['muted'])
        ax.set_xlim(1, 5)
        ax.tick_params(colors=DARK['muted'])
        for spine in ax.spines.values():
            spine.set_edgecolor(DARK['border'])
        ax.grid(True, color=DARK['border'], alpha=0.3, linestyle='--')
        ax.legend(fontsize=7)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Score Distribution per Construct',
                  color=DARK['text'], fontsize=13, fontweight='bold',
                  y=1.01)
    fig.tight_layout()
    save(fig, '05_distributions.png', out_dir)


# ─────────────────────────────────────────────────────────────────────────────
#  CHART 6 — Box Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_boxplots(scores: pd.DataFrame, out_dir: str):
    """Side-by-side box plots for all constructs."""
    fig, ax = plt.subplots(figsize=(max(8, len(scores.columns) * 1.4), 5),
                            facecolor=DARK['bg'])
    ax.set_facecolor(DARK['surface'])

    data_list = [scores[c].dropna().values for c in scores.columns]
    bp = ax.boxplot(data_list, patch_artist=True,
                    notch=False, vert=True,
                    medianprops=dict(color=DARK['text'], linewidth=2),
                    whiskerprops=dict(color=DARK['muted'], linewidth=1.2),
                    capprops=dict(color=DARK['muted'], linewidth=1.2),
                    flierprops=dict(marker='o', markerfacecolor=DARK['muted'],
                                    markersize=4, alpha=0.5))

    for patch, color in zip(bp['boxes'], PALETTE[:len(scores.columns)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Overlay individual data points (jitter)
    for i, (col, color) in enumerate(zip(scores.columns,
                                          PALETTE[:len(scores.columns)]), 1):
        y = scores[col].dropna().values
        x = np.random.normal(i, 0.06, len(y))
        ax.scatter(x, y, alpha=0.25, s=12, color=color,
                   edgecolors='none', zorder=3)

    ax.set_xticks(range(1, len(scores.columns) + 1))
    ax.set_xticklabels(scores.columns, fontsize=10)
    ax.set_ylabel('Score (Likert 1–5)')
    ax.set_ylim(0.5, 5.5)
    ax.set_title('Score Distribution — Box Plot with Data Points')
    ax.axhline(3.0, color=DARK['dim'], linewidth=1,
               linestyle=':', label='Neutral (3.0)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor(DARK['border'])
    fig.tight_layout()
    save(fig, '06_boxplots.png', out_dir)


# ─────────────────────────────────────────────────────────────────────────────
#  CHART 7 — Demographic Breakdown
# ─────────────────────────────────────────────────────────────────────────────

def plot_demographic(df: pd.DataFrame, scores: pd.DataFrame,
                      out_dir: str):
    """
    If demographic columns exist (Experience, Role, Org_Size etc.)
    plot mean OP/target score broken down by each group.
    """
    # Find non-numeric columns that could be demographics
    demo_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    target    = scores.columns[-1]  # Use last construct as target (usually OP)

    demo_cols = [c for c in demo_cols if df[c].nunique() <= 8]
    if not demo_cols:
        print("  ℹ️  No demographic columns found — skipping chart 7")
        return

    n_plots = min(3, len(demo_cols))
    fig, axes = plt.subplots(1, n_plots,
                              figsize=(n_plots * 5, 5),
                              facecolor=DARK['bg'])
    if n_plots == 1:
        axes = [axes]

    merged = df.copy()
    merged['__target__'] = scores[target].values

    for ax, demo in zip(axes, demo_cols[:n_plots]):
        ax.set_facecolor(DARK['surface'])
        groups    = merged.groupby(demo)['__target__'].mean().sort_values()
        bar_colors = PALETTE[:len(groups)]
        bars = ax.barh(groups.index.tolist(), groups.values,
                        color=bar_colors, edgecolor=DARK['bg'],
                        height=0.55, linewidth=0.8)
        for bar, val in zip(bars, groups.values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center', fontsize=9,
                    color=DARK['text'], fontweight='bold')
        ax.set_title(f'{target} by {demo}', color=DARK['text'])
        ax.set_xlabel('Mean Score')
        ax.set_xlim(0, 5.5)
        ax.axvline(3.0, color=DARK['dim'], linewidth=1, linestyle=':')
        ax.tick_params(colors=DARK['muted'])
        for spine in ax.spines.values():
            spine.set_edgecolor(DARK['border'])
        ax.grid(axis='x', alpha=0.3)

    fig.suptitle(f'Demographic Breakdown — {target}',
                  color=DARK['text'], fontsize=13, fontweight='bold')
    fig.tight_layout()
    save(fig, '07_demographic_breakdown.png', out_dir)


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY DASHBOARD — All charts in one figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_dashboard(df: pd.DataFrame, scores: pd.DataFrame,
                            constructs: dict, out_dir: str):
    """One combined 3×3 dashboard figure."""
    fig = plt.figure(figsize=(22, 16), facecolor=DARK['bg'])
    fig.suptitle(
        'Likert Scale Survey — Complete Analysis Dashboard\n'
        'Muhammad Hassaan Naeem · github.com/muhammad-hassaan-naeem',
        fontsize=14, fontweight='bold', color=DARK['text'], y=0.98
    )

    gs = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.45, wspace=0.35,
                            top=0.92, bottom=0.06,
                            left=0.07, right=0.97)

    means  = scores.mean()
    stds   = scores.std()
    n      = len(means)
    colors = PALETTE[:n]

    # ── Panel 1: Mean Scores Bar ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(DARK['surface'])
    bars = ax1.bar(means.index, means.values, yerr=stds.values,
                    color=colors, edgecolor=DARK['bg'],
                    linewidth=0.8, width=0.55,
                    error_kw=dict(ecolor=DARK['muted'], capsize=3))
    for bar, val in zip(bars, means.values):
        ax1.text(bar.get_x() + bar.get_width()/2,
                  val + 0.08, f'{val:.2f}',
                  ha='center', fontsize=8, color=DARK['text'],
                  fontweight='bold')
    ax1.set_ylim(0, 5.8)
    ax1.set_ylabel('Mean (1–5)')
    ax1.set_title('Mean Scores per Construct')
    ax1.axhline(3.0, color=DARK['dim'], linewidth=1, linestyle=':')
    ax1.tick_params(axis='x', rotation=20)
    for spine in ax1.spines.values():
        spine.set_edgecolor(DARK['border'])
    ax1.grid(axis='y', alpha=0.3)

    # ── Panel 2: Radar ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1], polar=True)
    ax2.set_facecolor(DARK['surface'])
    labels_r = list(means.index)
    N_r      = len(labels_r)
    vals_r   = list(means.values) + [means.values[0]]
    angs     = [n / N_r * 2 * np.pi for n in range(N_r)] + \
               [0 / N_r * 2 * np.pi]
    ax2.fill(angs, vals_r, alpha=0.2, color=DARK['green'])
    ax2.plot(angs, vals_r, 'o-', linewidth=2,
              color=DARK['green'], markersize=5)
    ax2.set_xticks(angs[:-1])
    ax2.set_xticklabels(labels_r, size=8, color=DARK['text'])
    ax2.set_ylim(1, 5)
    ax2.set_yticks([2, 3, 4, 5])
    ax2.set_yticklabels(['', '', '', ''], size=7)
    ax2.grid(color=DARK['border'], alpha=0.5)
    ax2.spines['polar'].set_color(DARK['border'])
    ax2.set_title('Radar Chart', color=DARK['text'], pad=15)

    # ── Panel 3: Box Plots ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(DARK['surface'])
    data_list = [scores[c].dropna().values for c in scores.columns]
    bp = ax3.boxplot(data_list, patch_artist=True,
                      medianprops=dict(color=DARK['text'], linewidth=1.8),
                      whiskerprops=dict(color=DARK['muted']),
                      capprops=dict(color=DARK['muted']),
                      flierprops=dict(marker='o', markerfacecolor=DARK['muted'],
                                      markersize=3, alpha=0.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_xticks(range(1, len(scores.columns) + 1))
    ax3.set_xticklabels(scores.columns, fontsize=8, rotation=20)
    ax3.set_ylim(0.5, 5.5)
    ax3.set_ylabel('Score')
    ax3.set_title('Box Plot Distribution')
    ax3.axhline(3.0, color=DARK['dim'], linewidth=1, linestyle=':')
    for spine in ax3.spines.values():
        spine.set_edgecolor(DARK['border'])
    ax3.grid(axis='y', alpha=0.3)

    # ── Panel 4: Heatmap ──────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    item_means = {}
    for name, items in constructs.items():
        for item in items:
            if item in df.columns:
                item_means[item] = df[item].mean()
    if item_means:
        im_df = pd.DataFrame(list(item_means.items()),
                              columns=['Item', 'Mean']).set_index('Item')
        cmap_h = LinearSegmentedColormap.from_list(
            'lk', ['#c0392b', '#f39c12', '#27ae60'], N=256)
        ax4.barh(im_df.index, im_df['Mean'],
                  color=[cmap_h((v-1)/4) for v in im_df['Mean']],
                  edgecolor=DARK['bg'], height=0.65)
        ax4.set_xlim(1, 5)
        ax4.axvline(3.0, color=DARK['dim'], linewidth=1, linestyle=':')
        ax4.set_xlabel('Mean Score')
        ax4.set_title('Item-Level Mean Scores')
        ax4.tick_params(axis='y', labelsize=7.5)
        ax4.set_facecolor(DARK['surface'])
        for spine in ax4.spines.values():
            spine.set_edgecolor(DARK['border'])
        ax4.grid(axis='x', alpha=0.3)

    # ── Panel 5: Response % stacked ───────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor(DARK['surface'])
    rc = response_counts(df, constructs)
    bottom = np.zeros(len(rc))
    resp_colors = [DIV_PALETTE[0], DIV_PALETTE[1], DIV_PALETTE[2],
                   DIV_PALETTE[3], DIV_PALETTE[4]]
    resp_labels = ['1 SD', '2 D', '3 N', '4 A', '5 SA']
    for col, color, lbl in zip([1,2,3,4,5], resp_colors, resp_labels):
        if col in rc.columns:
            ax5.bar(rc.index, rc[col].values, bottom=bottom,
                     color=color, edgecolor=DARK['bg'],
                     linewidth=0.5, width=0.55, label=lbl)
            bottom += rc[col].values
    ax5.set_ylabel('Percentage (%)')
    ax5.set_title('Stacked Response Distribution')
    ax5.legend(loc='upper right', fontsize=7, ncol=5)
    ax5.tick_params(axis='x', rotation=20)
    for spine in ax5.spines.values():
        spine.set_edgecolor(DARK['border'])
    ax5.grid(axis='y', alpha=0.3)

    # ── Panel 6: Correlation Heatmap ──────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    corr = scores.corr()
    cmap_c = sns.diverging_palette(10, 130, s=85, l=45, as_cmap=True)
    sns.heatmap(corr, ax=ax6, annot=True, fmt='.2f',
                cmap=cmap_c, center=0, vmin=-1, vmax=1,
                linewidths=0.6, linecolor=DARK['bg'],
                annot_kws={'size': 8, 'color': DARK['text']},
                cbar=False)
    ax6.set_facecolor(DARK['surface'])
    ax6.set_title('Correlation Matrix')
    ax6.tick_params(colors=DARK['muted'], rotation=30, labelsize=8)

    plt.savefig(os.path.join(out_dir, '00_dashboard.png'),
                dpi=150, bbox_inches='tight', facecolor=DARK['bg'])
    print("  ✅ Saved: 00_dashboard.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  PRINT SUMMARY STATS
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, scores: pd.DataFrame,
                   constructs: dict):
    print("\n" + "="*60)
    print("  SURVEY SUMMARY")
    print("="*60)
    print(f"  Respondents : {len(df)}")
    print(f"  Constructs  : {len(constructs)}")
    total_items = sum(len(v) for v in constructs.values())
    print(f"  Total items : {total_items}")
    print(f"\n  {'Construct':<12} {'Mean':>6}  {'Std':>6}  "
          f"{'Min':>5}  {'Max':>5}  {'Skew':>6}")
    print(f"  {'-'*12} {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*6}")
    for col in scores.columns:
        d = scores[col].dropna()
        print(f"  {col:<12} {d.mean():>6.3f}  {d.std():>6.3f}  "
              f"{d.min():>5.1f}  {d.max():>5.1f}  "
              f"{stats.skew(d):>6.3f}")
    print("="*60)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Likert Scale Survey Visualizer'
    )
    parser.add_argument('--file', '-f', required=True,
                        help='Path to CSV survey file')
    parser.add_argument('--constructs', '-c', default=None,
                        help='Path to JSON config file defining constructs')
    parser.add_argument('--output', '-o', default='output_charts',
                        help='Output directory for charts (default: output_charts)')
    parser.add_argument('--scale', '-s', type=int, default=5,
                        help='Likert scale size (default: 5)')
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    if not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}")
        sys.exit(1)

    print(f"\n📂 Loading: {args.file}")
    df = pd.read_csv(args.file)
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # ── Load or auto-detect constructs ────────────────────────────────────────
    if args.constructs and os.path.exists(args.constructs):
        with open(args.constructs) as f:
            constructs = json.load(f)
        print(f"📋 Constructs loaded from: {args.constructs}")
    else:
        print("🔍 Auto-detecting constructs from column names...")
        constructs = auto_detect_constructs(df)

    print(f"   Found {len(constructs)} construct(s): "
          f"{', '.join(constructs.keys())}")

    # ── Compute scores ────────────────────────────────────────────────────────
    scores = compute_scores(df, constructs)
    if scores.empty:
        print("❌ Could not compute construct scores. "
              "Check column names match your CSV.")
        sys.exit(1)

    # ── Output directory ──────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    print(f"\n📊 Generating charts → {args.output}/\n")

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary(df, scores, constructs)

    # ── Generate all charts ───────────────────────────────────────────────────
    print("\n  Generating individual charts:")
    plot_mean_scores(scores, args.output)
    plot_diverging_stacked(df, constructs, args.output)
    plot_radar(scores, args.output)
    plot_heatmap(df, constructs, args.output)
    plot_histograms(scores, args.output)
    plot_boxplots(scores, args.output)
    plot_demographic(df, scores, args.output)

    print("\n  Generating summary dashboard:")
    plot_summary_dashboard(df, scores, constructs, args.output)

    print(f"\n🎉 Done! {args.output}/ contains 8 chart files.")
    print(f"   Open 00_dashboard.png for the full overview.\n")


if __name__ == '__main__':
    main()
