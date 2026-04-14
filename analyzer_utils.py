"""Utility helpers for TeamAnalyzer data parsing and normalization."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

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


def build_api_headers(api_key):
    """Return the standard SportsData.io request headers dict."""
    return {'Accept': 'application/json', 'Ocp-Apim-Subscription-Key': api_key}


def is_nan(value):
    """Return True if value is None or a float NaN."""
    return value is None or (isinstance(value, float) and np.isnan(value))


def load_env_file(env_path='.env'):
    """Load KEY=VALUE pairs from a .env file into the environment."""
    if not os.path.exists(env_path):
        return
    with open(env_path, 'r', encoding='utf-8') as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def extract_api_key_from_url(url):
    """Strip the 'key' query param from a URL; return (clean_url, key)."""
    if not url:
        return url, None
    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    values = query.pop('key', None)
    extracted_key = values[0] if values else None
    cleaned_query = urlencode(query, doseq=True)
    cleaned_url = urlunparse((
        parsed.scheme, parsed.netloc, parsed.path,
        parsed.params, cleaned_query, parsed.fragment,
    ))
    return cleaned_url, extracted_key


def pick_value(item, *keys):
    """Return the first non-empty value for any of the candidate keys."""
    for key in keys:
        value = item.get(key)
        if value is not None and value != '':
            return value
    return np.nan


def team_matches(row_team_name, requested_team_name):
    """Case-insensitive exact match between two team name strings."""
    if pd.isna(row_team_name):
        return False
    return (
        str(row_team_name).strip().lower()
        == str(requested_team_name).strip().lower()
    )


def build_season_row(team_stats, team_name):
    """Extract and return a normalised season-row dict from a raw stats dict."""
    goals_scored = pick_value(team_stats, 'Score', 'Goals', 'GoalsScored', 'TeamGoals')
    goals_conceded = pick_value(
        team_stats, 'OpponentScore', 'OpponentGoals', 'GoalsAgainst', 'GoalsConceded'
    )
    possession = pick_value(team_stats, 'Possession', 'PossessionPct', 'PossessionPercentage')
    shots = pick_value(team_stats, 'Shots', 'ShotsTotal', 'TeamShots')
    shots_on_target = pick_value(team_stats, 'ShotsOnGoal', 'ShotsOnTarget')
    matches_in_sample = pick_value(
        team_stats, 'Games', 'GamesPlayed', 'Matches', 'MatchesPlayed'
    )
    clean_sheets = pick_value(team_stats, *_PV)
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
        'goals_scored': goals_scored,
        'goals_conceded': goals_conceded,
        'possession': possession,
        'shots': shots,
        'shots_on_target': shots_on_target,
        'matches_in_sample': matches_in_sample,
        'clean_sheets': clean_sheets,
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

    def _coerce_dict(data):
        for key in ('TeamSeasonStats', 'teamSeasonStats', 'data', 'Data'):
            value = data.get(key)
            if isinstance(value, list):
                return _coerce(value)
            if isinstance(value, dict):
                return _coerce([value])
        for key in ('Rounds', 'rounds'):
            rounds = data.get(key)
            if isinstance(rounds, list):
                return _coerce(rounds)
        if isinstance(data.get('TeamSeasons'), list):
            context = {
                'Season': data.get('Season'),
                'SeasonType': data.get('SeasonType'),
                'RoundId': data.get('RoundId'),
                'RoundName': data.get('Name'),
            }
            return _coerce(data['TeamSeasons'], round_context=context)
        return _coerce([data])

    if isinstance(payload, list):
        return _coerce(payload)
    if isinstance(payload, dict):
        return _coerce_dict(payload)
    return []


def fetch_json(url, headers=None, params=None):
    """Fetch and return a parsed JSON payload from an HTTP endpoint."""
    if params:
        separator = '&' if '?' in url else '?'
        url = f"{url}{separator}{urlencode(params)}"
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=30) as response:
        payload = response.read().decode('utf-8', errors='ignore')
    return json.loads(payload)


def log_raw_api_data(season_year, raw_rows, team_name, log_path=None, existing_buffer=None):
    """Write raw API rows for a season to a timestamped JSON log file."""
    if log_path is None:
        safe_name = team_name.lower().replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(__file__).parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f'{safe_name}_api_data_{timestamp}.json'
    else:
        log_path = Path(log_path)

    existing = existing_buffer or {}
    existing[str(season_year)] = {
        'fetched_at': datetime.now().isoformat(timespec='seconds'),
        'season_year': season_year,
        'team': team_name,
        'row_count': len(raw_rows),
        'data': raw_rows,
    }
    log_path.write_text(
        json.dumps({'api_log': existing}, indent=2, default=str),
        encoding='utf-8',
    )
    return existing
