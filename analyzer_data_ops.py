"""Operational data-loading/fetching functions for TeamAnalyzer."""

import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from urllib.error import URLError

import pandas as pd

from analyzer_http import build_api_headers
from analyzer_logging import log_raw_api_data
from analyzer_payloads import build_season_row, coerce_numeric_cols, is_nan


def extract_team_season_row(analyzer, team_stats, team_name='Arsenal FC'):
    """Convert a raw TeamSeasonStats dict into a normalized row dict."""
    source_team = analyzer._pick_value(team_stats, 'Name', 'TeamName', 'Team', 'TeamKey')
    if not analyzer._team_matches(source_team, team_name):
        return None
    return build_season_row(team_stats, team_name)


def load_match_data_from_api(analyzer, api_url, api_key, team_name, log_path=None):
    """Fetch and parse TeamSeason rows for one season URL."""
    cleaned_url, key_from_url = analyzer._extract_api_key_from_url(api_url)
    resolved_api_key = api_key or key_from_url
    if not resolved_api_key:
        raise ValueError('A SportsData.io API key is required.')

    payload = analyzer._fetch_json(cleaned_url, headers=build_api_headers(resolved_api_key))
    normalized = analyzer._normalize_team_season_payload(payload)

    raw_team_rows = [
        item for item in normalized
        if isinstance(item, dict)
        and analyzer._team_matches(analyzer._pick_value(item, 'Name', 'TeamName', 'Team', 'TeamKey'), team_name)
    ]

    rows = [
        row for row in (extract_team_season_row(analyzer, item, team_name=team_name) for item in normalized)
        if row
    ]
    if not rows:
        raise ValueError(f'No {team_name} data parsed from: {cleaned_url}')

    existing = getattr(analyzer.__class__, '_log_buffer', {})
    analyzer.__class__._log_buffer = log_raw_api_data(
        rows[0].get('season_year', 'unknown'),
        raw_team_rows,
        team_name,
        log_path=log_path,
        existing_buffer=existing,
    )

    data_frame = pd.DataFrame(rows)
    coerce_numeric_cols(data_frame)
    return data_frame


def load_sample_data(analyzer, api_key, seasons, team_name):
    """Load data for each season and combine into analyzer.match_data."""
    analyzer._load_env_file()
    analyzer.__class__._log_buffer = {}
    resolved_api_key = api_key or os.getenv('SPORTSDATA_API_KEY')
    if not resolved_api_key:
        warnings.warn('SPORTSDATA_API_KEY is not configured. Using fallback sample data.', RuntimeWarning)
        analyzer.match_data = analyzer._default_match_data(team_name=team_name)
        return

    safe_name = team_name.lower().replace(' ', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    run_log_path = log_dir / f'{safe_name}_api_data_{timestamp}.json'

    season_frames = []
    for season_year in seasons:
        try:
            season_frames.append(
                load_match_data_from_api(
                    analyzer,
                    analyzer._url_for_season(season_year),
                    resolved_api_key,
                    team_name,
                    log_path=run_log_path,
                )
            )
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            warnings.warn(f'Season {season_year}: failed to load ({exc}). Skipping.', RuntimeWarning)

    if not season_frames:
        warnings.warn('All seasons failed to load. Using fallback sample data.', RuntimeWarning)
        analyzer.match_data = analyzer._default_match_data(team_name=team_name)
        return

    analyzer.match_data = pd.concat(season_frames, ignore_index=True).sort_values('date').reset_index(drop=True)


def fetch_competition_details(analyzer, api_key=None, competition_id=3):
    """Return a dict of {team_name: wikipedia_logo_url} for a competition."""
    analyzer._load_env_file()
    resolved_api_key = api_key or os.getenv('SPORTSDATA_API_KEY')
    if not resolved_api_key:
        warnings.warn('SPORTSDATA_API_KEY is not configured. Cannot fetch competition details.', RuntimeWarning)
        return {}

    url = analyzer.competition_details_template.format(competition_id=competition_id)
    try:
        payload = analyzer._fetch_json(url, headers=build_api_headers(resolved_api_key))
    except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        warnings.warn(f'Failed to fetch competition details ({exc}). No logos will be shown.', RuntimeWarning)
        return {}

    logos = {}
    for team in payload.get('Teams') or payload.get('teams') or []:
        if not isinstance(team, dict):
            continue
        name = team.get('Name') or team.get('TeamName') or team.get('ShortName')
        if name:
            logos[str(name).strip()] = team.get('WikipediaLogoUrl') or team.get('wikipediaLogoUrl') or None
    return logos


def fetch_team_names_for_season(analyzer, season_year, api_key):
    """Fetch and return team names for a specific season."""
    cleaned_url, key_from_url = analyzer._extract_api_key_from_url(analyzer._url_for_season(season_year))
    payload = analyzer._fetch_json(cleaned_url, headers=build_api_headers(api_key or key_from_url))
    rows = analyzer._normalize_team_season_payload(payload)

    names = set()
    for item in rows:
        if not isinstance(item, dict):
            continue
        name = analyzer._pick_value(item, 'Name', 'TeamName', 'Team', 'TeamKey')
        if not is_nan(name):
            names.add(str(name).strip())
    return names


def fetch_all_teams(analyzer, api_key, seasons):
    """Return sorted, deduplicated team names from the API."""
    analyzer._load_env_file()
    resolved_api_key = api_key or os.getenv('SPORTSDATA_API_KEY')
    if not resolved_api_key:
        warnings.warn('SPORTSDATA_API_KEY is not configured. Returning fallback team list.', RuntimeWarning)
        return [analyzer.default_team_name]

    team_names = set()
    for season_year in seasons:
        try:
            team_names.update(fetch_team_names_for_season(analyzer, season_year, resolved_api_key))
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            warnings.warn(f'Season {season_year}: could not fetch team list ({exc}). Skipping.', RuntimeWarning)

    if not team_names:
        warnings.warn('No teams found from API. Returning fallback team list.', RuntimeWarning)
        return [analyzer.default_team_name]
    return sorted(team_names)


def fetch_league_possession_averages(analyzer, api_key, seasons):
    """Fetch team possession averages across provided seasons."""
    analyzer._load_env_file()
    resolved_api_key = api_key or os.getenv('SPORTSDATA_API_KEY')
    if not resolved_api_key:
        if analyzer.match_data is not None and not analyzer.match_data.empty:
            return analyzer.match_data.groupby('team', as_index=True)['possession'].mean().dropna().to_dict()
        return {analyzer.default_team_name: 56.8}

    season_frames = []
    for season_year in seasons:
        cleaned_url, key_from_url = analyzer._extract_api_key_from_url(analyzer._url_for_season(season_year))
        try:
            payload = analyzer._fetch_json(cleaned_url, headers=build_api_headers(resolved_api_key or key_from_url))
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError):
            continue

        rows = analyzer._normalize_team_season_payload(payload)
        season_rows = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            team_name = analyzer._pick_value(item, 'Name', 'TeamName', 'Team', 'TeamKey')
            possession = analyzer._pick_value(item, 'Possession', 'PossessionPct', 'PossessionPercentage')
            if is_nan(team_name) or is_nan(possession):
                continue
            season_rows.append({'team': str(team_name).strip(), 'possession': possession})

        if season_rows:
            frame = pd.DataFrame(season_rows)
            coerce_numeric_cols(frame)
            season_frames.append(frame)

    if not season_frames:
        return {analyzer.default_team_name: 56.8}

    combined = pd.concat(season_frames, ignore_index=True)
    return combined.groupby('team', as_index=True)['possession'].mean().dropna().to_dict()
