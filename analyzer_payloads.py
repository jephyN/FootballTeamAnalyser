"""Payload normalization and dataframe utility helpers."""

import numpy as np
import pandas as pd

_PV = "GoalkeeperCleanSheets", "DefenderCleanSheets", "CleanSheets"

NUMERIC_COLS = [
    'goals_scored', 'goals_conceded', 'possession',
    'shots', 'shots_on_target', 'matches_in_sample', 'clean_sheets',
]


def coerce_numeric_cols(data_frame):
    """Coerce known numeric columns to float in-place."""
    for col in NUMERIC_COLS:
        if col in data_frame.columns:
            data_frame[col] = pd.to_numeric(data_frame[col], errors='coerce')


def is_nan(value):
    """Return True if value is None or a float NaN."""
    return value is None or (isinstance(value, float) and np.isnan(value))


def pick_value(item, *keys):
    """Return the first non-empty value for any candidate keys."""
    for key in keys:
        value = item.get(key)
        if value is not None and value != '':
            return value
    return np.nan


def team_matches(row_team_name, requested_team_name):
    """Case-insensitive exact match between two team names."""
    if pd.isna(row_team_name):
        return False
    return str(row_team_name).strip().lower() == str(requested_team_name).strip().lower()


def build_season_row(team_stats, team_name):
    """Extract and return a normalised season-row dict from raw stats."""
    season = pick_value(team_stats, 'Season', 'SeasonYear')
    season_year = pd.to_numeric(season, errors='coerce')
    season_date = (
        pd.to_datetime(f"{int(season_year)}-12-31", errors='coerce')
        if not pd.isna(season_year) else pd.NaT
    )
    return {
        'date': season_date,
        'season_year': int(season_year) if not pd.isna(season_year) else np.nan,
        'team': team_name,
        'opponent': 'All Teams (Season Aggregate)',
        'venue': 'Season',
        'goals_scored': pick_value(team_stats, 'Score', 'Goals', 'GoalsScored', 'TeamGoals'),
        'goals_conceded': pick_value(team_stats, 'OpponentScore', 'OpponentGoals', 'GoalsAgainst', 'GoalsConceded'),
        'possession': pick_value(team_stats, 'Possession', 'PossessionPct', 'PossessionPercentage'),
        'shots': pick_value(team_stats, 'Shots', 'ShotsTotal', 'TeamShots'),
        'shots_on_target': pick_value(team_stats, 'ShotsOnGoal', 'ShotsOnTarget'),
        'matches_in_sample': pick_value(team_stats, 'Games', 'GamesPlayed', 'Matches', 'MatchesPlayed'),
        'clean_sheets': pick_value(team_stats, *_PV),
        'season_type': pick_value(team_stats, 'SeasonType'),
        'team_id': pick_value(team_stats, 'TeamId'),
    }


def normalize_team_season_payload(payload):
    """Normalise TeamSeasonStats/Round payloads to a flat row list."""
    def _coerce(items, round_context=None):
        rows = []
        for item in items:
            if not isinstance(item, dict):
                continue
            team_seasons = item.get('TeamSeasons')
            if isinstance(team_seasons, list):
                context = {
                    'Season': item.get('Season'),
                    'SeasonType': item.get('SeasonType'),
                    'RoundId': item.get('RoundId'),
                    'RoundName': item.get('Name'),
                }
                for team_row in team_seasons:
                    if not isinstance(team_row, dict):
                        continue
                    merged = dict(team_row)
                    for key, value in context.items():
                        if merged.get(key) in (None, '') and value not in (None, ''):
                            merged[key] = value
                    rows.append(merged)
            else:
                merged = dict(item)
                if round_context:
                    for key, value in round_context.items():
                        if merged.get(key) in (None, '') and value not in (None, ''):
                            merged[key] = value
                rows.append(merged)
        return rows

    if isinstance(payload, list):
        return _coerce(payload)
    if not isinstance(payload, dict):
        return []

    for key in ('TeamSeasonStats', 'teamSeasonStats', 'data', 'Data'):
        value = payload.get(key)
        if isinstance(value, list):
            return _coerce(value)
        if isinstance(value, dict):
            return _coerce([value])

    for key in ('Rounds', 'rounds'):
        rounds = payload.get(key)
        if isinstance(rounds, list):
            return _coerce(rounds)

    if isinstance(payload.get('TeamSeasons'), list):
        context = {
            'Season': payload.get('Season'),
            'SeasonType': payload.get('SeasonType'),
            'RoundId': payload.get('RoundId'),
            'RoundName': payload.get('Name'),
        }
        return _coerce(payload['TeamSeasons'], round_context=context)

    return _coerce([payload])
