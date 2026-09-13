"""Statistical reporting helpers for TeamAnalyzer."""

from datetime import datetime

import numpy as np
import pandas as pd

from analyzer_payloads import is_nan


def calculate_basic_stats(match_data, team_name, season_year=None):
    """Return per-round average possession and shot accuracy for a team."""
    mask = match_data['team'] == team_name
    if season_year is not None and 'season_year' in match_data.columns:
        mask = mask & (match_data['season_year'] == season_year)
    rounds = match_data[mask]

    avg_possession = rounds['possession'].mean() if 'possession' in rounds.columns else np.nan

    if {'shots_on_target', 'shots'}.issubset(rounds.columns):
        valid = rounds[rounds['shots'].notna() & (rounds['shots'] > 0)].copy()
        shot_accuracy = (valid['shots_on_target'] / valid['shots'] * 100).mean() if not valid.empty else np.nan
    else:
        shot_accuracy = np.nan

    return pd.Series({
        'avg_possession': round(avg_possession, 2) if not pd.isna(avg_possession) else np.nan,
        'shot_accuracy': round(shot_accuracy, 2) if not pd.isna(shot_accuracy) else np.nan,
    }, dtype='object')


def generate_report(analyzer, team_name, seasons):
    """Return a side-by-side text report of metrics across seasons."""
    season_stats = {}
    available_seasons = []
    for yr in seasons:
        if 'season_year' in analyzer.match_data.columns and yr not in analyzer.match_data['season_year'].values:
            continue
        season_stats[yr] = analyzer.calculate_basic_stats(team_name, season_year=yr)
        available_seasons.append(yr)

    if not available_seasons:
        return f'No data available for {team_name} across seasons {seasons}.'

    label_w = 24
    col_w = 14
    header_seasons = ''.join(str(yr).rjust(col_w) for yr in available_seasons)
    separator = '-' * (label_w + col_w * len(available_seasons))

    def row(label, key, spec='.2f', suffix=''):
        values = ''
        for year in available_seasons:
            raw = season_stats[year].get(key, np.nan)
            cell = analyzer._fmt(raw, spec) + suffix if not is_nan(raw) else 'N/A'
            values += cell.rjust(col_w)
        return f'{label:<{label_w}}{values}'

    lines = [
        f'\nPerformance Report — {team_name}',
        f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '(Values are averages per round across each season)',
        '',
        f'{"Season":<{label_w}}{header_seasons}',
        separator,
        'Performance Metrics:',
        row('  Avg Possession', 'avg_possession', '.2f', '%'),
        row('  Shot Accuracy', 'shot_accuracy', '.2f', '%'),
        separator,
    ]
    return '\n'.join(lines)
