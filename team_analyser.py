"""Main analyzer API for football team data."""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pandas as pd

from analyzer_plotting import plot_performance_trends as _plot_performance_trends
from analyzer_utils import (
    build_api_headers,
    build_season_row,
    coerce_numeric_cols,
    extract_api_key_from_url,
    fetch_json,
    is_nan,
    load_env_file,
    log_raw_api_data,
    normalize_team_season_payload,
    pick_value,
    team_matches,
)
from possession_prediction import build_predictions
from logo_utils import _fetch_logo_pil


class TeamAnalyzer:
    """Handles data acquisition, normalization, analytics, and reporting."""

    api_url_template = (
        'https://api.sportsdata.io/v4/soccer/scores/json/TeamSeasonStats/3/{season}'
    )
    competition_details_template = (
        'https://api.sportsdata.io/v4/soccer/scores/json/CompetitionDetails/{competition_id}'
    )
    default_seasons = (2025, 2026)
    default_team_name = 'Arsenal FC'

    def __init__(self):
        self.team_data = None
        self.match_data = None

    @staticmethod
    def _fmt(value, spec='.2f'):
        if is_nan(value):
            return 'N/A'
        try:
            return format(value, spec)
        except (TypeError, ValueError):
            return 'N/A'

    @staticmethod
    def _default_match_data(team_name='Arsenal FC'):
        return pd.DataFrame({
            'date': pd.to_datetime(['2025-12-31', '2026-12-31']),
            'season_year': [2025, 2026],
            'team': [team_name] * 2,
            'possession': [56.8, 57.5],
            'shots': [495, 518],
            'shots_on_target': [172, 189],
        })

    @staticmethod
    def _fetch_json(url, headers=None, params=None):
        return fetch_json(url, headers=headers, params=params)

    @staticmethod
    def _normalize_team_season_payload(payload):
        return normalize_team_season_payload(payload)

    @staticmethod
    def _load_env_file(env_path='.env'):
        load_env_file(env_path=env_path)

    @staticmethod
    def _extract_api_key_from_url(url):
        return extract_api_key_from_url(url)

    @staticmethod
    def _pick_value(item, *keys):
        return pick_value(item, *keys)

    @staticmethod
    def _team_matches(row_team_name, requested_team_name):
        return team_matches(row_team_name, requested_team_name)

    @staticmethod
    def _extract_team_season_row(team_stats, team_name='Arsenal FC'):
        source_team = TeamAnalyzer._pick_value(
            team_stats, 'Name', 'TeamName', 'Team', 'TeamKey'
        )
        if not TeamAnalyzer._team_matches(source_team, team_name):
            return None
        return build_season_row(team_stats, team_name)

    @classmethod
    def _url_for_season(cls, season_year):
        return cls.api_url_template.format(season=season_year)

    @staticmethod
    def _log_raw_api_data(season_year, raw_rows, team_name, log_path=None):
        existing = getattr(TeamAnalyzer, '_log_buffer', {})
        TeamAnalyzer._log_buffer = log_raw_api_data(
            season_year, raw_rows, team_name, log_path=log_path, existing_buffer=existing
        )

    def load_match_data_from_api(
        self, api_url, api_key, team_name=default_team_name, log_path=None
    ):
        cleaned_url, key_from_url = self._extract_api_key_from_url(api_url)
        resolved_api_key = api_key or key_from_url
        if not resolved_api_key:
            raise ValueError('A SportsData.io API key is required.')
        headers = build_api_headers(resolved_api_key)
        payload = self._fetch_json(cleaned_url, headers=headers)
        normalized = self._normalize_team_season_payload(payload)
        raw_team_rows = [
            item for item in normalized
            if isinstance(item, dict)
            and self._team_matches(
                self._pick_value(item, 'Name', 'TeamName', 'Team', 'TeamKey'),
                team_name,
            )
        ]
        rows = [
            row for row in (
                self._extract_team_season_row(r, team_name=team_name)
                for r in normalized
            )
            if row
        ]
        if not rows:
            raise ValueError(f'No {team_name} data parsed from: {cleaned_url}')
        season_year = rows[0].get('season_year', 'unknown')
        self._log_raw_api_data(season_year, raw_team_rows, team_name=team_name, log_path=log_path)
        data_frame = pd.DataFrame(rows)
        coerce_numeric_cols(data_frame)
        return data_frame

    def load_sample_data(
        self, api_key=None, seasons=default_seasons, team_name=default_team_name
    ):
        self._load_env_file()
        TeamAnalyzer._log_buffer = {}
        resolved_api_key = api_key or os.getenv('SPORTSDATA_API_KEY')
        if not resolved_api_key:
            warnings.warn(
                'SPORTSDATA_API_KEY is not configured. Using fallback sample data.',
                RuntimeWarning,
            )
            self.match_data = self._default_match_data(team_name=team_name)
            return

        safe_name = team_name.lower().replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(__file__).parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        run_log_path = log_dir / f'{safe_name}_api_data_{timestamp}.json'

        season_frames = []
        for season_year in seasons:
            url = self._url_for_season(season_year)
            try:
                data_frame = self.load_match_data_from_api(
                    api_url=url,
                    api_key=resolved_api_key,
                    team_name=team_name,
                    log_path=run_log_path,
                )
                season_frames.append(data_frame)
            except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                warnings.warn(
                    f'Season {season_year}: failed to load ({exc}). Skipping.',
                    RuntimeWarning,
                )
        if not season_frames:
            warnings.warn(
                'All seasons failed to load. Using fallback sample data.',
                RuntimeWarning,
            )
            self.match_data = self._default_match_data(team_name=team_name)
            return
        combined = pd.concat(season_frames, ignore_index=True)
        self.match_data = combined.sort_values('date').reset_index(drop=True)

    def calculate_basic_stats(self, team_name, season_year=None):
        mask = self.match_data['team'] == team_name
        if season_year is not None and 'season_year' in self.match_data.columns:
            mask = mask & (self.match_data['season_year'] == season_year)
        rounds = self.match_data[mask]
        avg_possession = rounds['possession'].mean() if 'possession' in rounds.columns else np.nan

        if {'shots_on_target', 'shots'}.issubset(rounds.columns):
            valid = rounds[rounds['shots'].notna() & (rounds['shots'] > 0)].copy()
            if not valid.empty:
                valid['round_accuracy'] = valid['shots_on_target'] / valid['shots'] * 100
                shot_accuracy = valid['round_accuracy'].mean()
            else:
                shot_accuracy = np.nan
        else:
            shot_accuracy = np.nan

        return pd.Series({
            'avg_possession': round(avg_possession, 2) if not pd.isna(avg_possession) else np.nan,
            'shot_accuracy': round(shot_accuracy, 2) if not pd.isna(shot_accuracy) else np.nan,
        }, dtype='object')

    def fetch_competition_details(self, api_key=None, competition_id=3):
        self._load_env_file()
        resolved_api_key = api_key or os.getenv('SPORTSDATA_API_KEY')
        if not resolved_api_key:
            warnings.warn(
                'SPORTSDATA_API_KEY is not configured. Cannot fetch competition details.',
                RuntimeWarning,
            )
            return {}

        url = self.competition_details_template.format(competition_id=competition_id)
        headers = build_api_headers(resolved_api_key)
        try:
            payload = self._fetch_json(url, headers=headers)
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            warnings.warn(
                f'Failed to fetch competition details ({exc}). No logos will be shown.',
                RuntimeWarning,
            )
            return {}

        logos = {}
        teams = payload.get('Teams') or payload.get('teams') or []
        for team in teams:
            if not isinstance(team, dict):
                continue
            name = team.get('Name') or team.get('TeamName') or team.get('ShortName')
            logo_url = team.get('WikipediaLogoUrl') or team.get('wikipediaLogoUrl')
            if name:
                logos[str(name).strip()] = logo_url or None
        return logos

    def fetch_all_teams(self, api_key=None, seasons=default_seasons):
        self._load_env_file()
        resolved_api_key = api_key or os.getenv('SPORTSDATA_API_KEY')
        if not resolved_api_key:
            warnings.warn(
                'SPORTSDATA_API_KEY is not configured. Returning fallback team list.',
                RuntimeWarning,
            )
            return [self.default_team_name]

        team_names = set()
        for season_year in seasons:
            try:
                team_names.update(self._fetch_team_names_for_season(season_year, resolved_api_key))
            except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                warnings.warn(
                    f'Season {season_year}: could not fetch team list ({exc}). Skipping.',
                    RuntimeWarning,
                )
        if not team_names:
            warnings.warn(
                'No teams found from API. Returning fallback team list.',
                RuntimeWarning,
            )
            return [self.default_team_name]
        return sorted(team_names)

    def _fetch_team_names_for_season(self, season_year, api_key):
        url = self._url_for_season(season_year)
        cleaned_url, key_from_url = self._extract_api_key_from_url(url)
        headers = build_api_headers(api_key or key_from_url)
        payload = self._fetch_json(cleaned_url, headers=headers)
        rows = self._normalize_team_season_payload(payload)

        team_names = set()
        for item in rows:
            if not isinstance(item, dict):
                continue
            name = self._pick_value(item, 'Name', 'TeamName', 'Team', 'TeamKey')
            if not is_nan(name):
                team_names.add(str(name).strip())
        return team_names

    def fetch_league_possession_averages(self, api_key=None, seasons=default_seasons):
        """Fetch team possession averages across provided seasons."""
        self._load_env_file()
        resolved_api_key = api_key or os.getenv('SPORTSDATA_API_KEY')
        if not resolved_api_key:
            if self.match_data is not None and not self.match_data.empty:
                return (
                    self.match_data.groupby('team', as_index=True)['possession']
                    .mean()
                    .dropna()
                    .to_dict()
                )
            return {self.default_team_name: 56.8}

        season_frames = []
        for season_year in seasons:
            url = self._url_for_season(season_year)
            cleaned_url, key_from_url = self._extract_api_key_from_url(url)
            headers = build_api_headers(resolved_api_key or key_from_url)
            try:
                payload = self._fetch_json(cleaned_url, headers=headers)
            except (URLError, TimeoutError, ValueError, json.JSONDecodeError):
                continue
            rows = self._normalize_team_season_payload(payload)
            season_rows = []
            for item in rows:
                if not isinstance(item, dict):
                    continue
                team_name = self._pick_value(item, 'Name', 'TeamName', 'Team', 'TeamKey')
                possession = self._pick_value(item, 'Possession', 'PossessionPct', 'PossessionPercentage')
                if is_nan(team_name) or is_nan(possession):
                    continue
                season_rows.append({'team': str(team_name).strip(), 'possession': possession})
            if season_rows:
                frame = pd.DataFrame(season_rows)
                coerce_numeric_cols(frame)
                season_frames.append(frame)

        if not season_frames:
            return {self.default_team_name: 56.8}

        combined = pd.concat(season_frames, ignore_index=True)
        grouped = combined.groupby('team', as_index=True)['possession'].mean().dropna()
        return grouped.to_dict()

    def predict_possession_vs_opponents(self, team_name, opponents, api_key=None, seasons=default_seasons):
        """Predict possession percentages for team vs opponent list."""
        team_possessions = self.fetch_league_possession_averages(api_key=api_key, seasons=seasons)
        return build_predictions(team_name, opponents, team_possessions)

    def generate_report(self, team_name, seasons=default_seasons):
        season_stats = {}
        available_seasons = []
        for yr in seasons:
            if (
                'season_year' in self.match_data.columns
                and yr not in self.match_data['season_year'].values
            ):
                continue
            season_stats[yr] = self.calculate_basic_stats(team_name, season_year=yr)
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
                cell = self._fmt(raw, spec) + suffix if not is_nan(raw) else 'N/A'
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

    def plot_performance_trends(self, team_name, seasons=default_seasons, logo_url=None):
        return _plot_performance_trends(
            self,
            team_name,
            seasons,
            logo_url=logo_url,
            fetch_logo_fn=_fetch_logo_pil,
        )


# Backward-compatible exports used in tests
_build_api_headers = build_api_headers
_build_season_row = lambda team_stats, pick, team_name: build_season_row(team_stats, team_name)
_coerce_numeric_cols = coerce_numeric_cols
_is_nan = is_nan
