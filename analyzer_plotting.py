"""Plotting helpers for TeamAnalyzer charts."""

import matplotlib.pyplot as plt
import numpy as np

from logo_utils import _fetch_logo_pil
from analyzer_payloads import is_nan


def collect_season_stats(analyzer, team_name, seasons):
    """Return (available_seasons, possession_vals, shot_accuracy_vals) lists."""
    available_seasons, possession_vals, shot_accuracy_vals = [], [], []
    for yr in seasons:
        if (
            'season_year' in analyzer.match_data.columns
            and yr not in analyzer.match_data['season_year'].values
        ):
            continue
        stats = analyzer.calculate_basic_stats(team_name, season_year=yr)
        available_seasons.append(yr)
        possession_vals.append(stats.get('avg_possession', np.nan))
        shot_accuracy_vals.append(stats.get('shot_accuracy', np.nan))
    return available_seasons, possession_vals, shot_accuracy_vals


def add_chart_logo(fig, logo_img, title_top):
    """Overlay the team logo in the header strip; return title x offset."""
    logo_ax = fig.add_axes([0.01, title_top + 0.005, 0.08, 1.0 - title_top - 0.01])
    logo_ax.imshow(np.array(logo_img))
    logo_ax.axis('off')
    return 0.54


def annotate_line(ax, labels, values, fmt='.2f', suffix='%'):
    """Annotate each point on a line chart with its formatted value."""
    for lbl, val in zip(labels, values):
        if not is_nan(val):
            ax.annotate(
                f'{val:{fmt}}{suffix}', (lbl, val),
                textcoords='offset points', xytext=(0, 8),
                ha='center', fontsize=9,
            )


def plot_metric_axis(ax, labels, values, style):
    """Plot one metric trend line or a no-data placeholder on the given axis."""
    if any(not is_nan(v) for v in values):
        ax.plot(
            labels, values, marker=style['marker'], color=style['color'],
            linewidth=2, markersize=8, label=style['label'],
        )
        annotate_line(ax, labels, values)
        ax.set_title(style['title'])
        ax.set_ylabel(style['ylabel'])
        ax.set_ylim(0, 100)
    else:
        ax.text(
            0.5, 0.5, style['no_data_msg'], ha='center', va='center', transform=ax.transAxes
        )
        ax.set_axis_off()
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)


def plot_performance_trends(analyzer, team_name, seasons, logo_url=None, fetch_logo_fn=_fetch_logo_pil):
    """Return a matplotlib figure with possession and shot accuracy trends."""
    available_seasons, possession_vals, shot_accuracy_vals = (
        collect_season_stats(analyzer, team_name, seasons)
    )
    season_labels = [str(yr) for yr in available_seasons]
    logo_img = fetch_logo_fn(logo_url, size=(52, 52)) if logo_url else None
    has_logo = logo_img is not None
    title_top = 0.86

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    plot_metric_axis(
        ax1, season_labels, possession_vals, {
            'marker': 'o',
            'color': 'green',
            'label': 'Avg Possession %',
            'title': 'Average Possession % per Round',
            'ylabel': 'Possession (%)',
            'no_data_msg': 'Possession data not available',
        }
    )
    plot_metric_axis(
        ax2, season_labels, shot_accuracy_vals, {
            'marker': 's',
            'color': 'darkorange',
            'label': 'Avg Shot Accuracy %',
            'title': 'Average Shot Accuracy % per Round',
            'ylabel': 'Shot Accuracy (%)',
            'no_data_msg': 'Shot accuracy data not available',
        }
    )

    plt.tight_layout(rect=[0, 0, 1, title_top])

    title_x = add_chart_logo(fig, logo_img, title_top) if has_logo else 0.5

    fig.suptitle(
        f'{team_name} — Season-by-Season Performance Metrics',
        fontsize=14, fontweight='bold',
        y=(title_top + 1.0) / 2,
        x=title_x,
    )
    return fig
