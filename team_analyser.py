"""Main analyzer API for football team data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer_data_ops import (
    fetch_all_teams as _fetch_all_teams,
    fetch_competition_details as _fetch_competition_details,
    fetch_league_possession_averages as _fetch_league_possession_averages,
    fetch_team_names_for_season as _fetch_team_names_for_season,
    load_match_data_from_api as _load_match_data_from_api,
    load_sample_data as _load_sample_data,
)
from analyzer_logging import log_raw_api_data
from analyzer_http import (
    build_api_headers,
    extract_api_key_from_url,
    fetch_json,
    load_env_file,
)
from analyzer_payloads import (
    build_season_row,
    coerce_numeric_cols,
    is_nan,
    normalize_team_season_payload,
    pick_value,
    team_matches,
)
from analyzer_plotting import plot_performance_trends as _plot_performance_trends
from analyzer_reporting import calculate_basic_stats as _calculate_basic_stats
from analyzer_reporting import generate_report as _generate_report
from logo_utils import _fetch_logo_pil
from possession_prediction import build_predictions


class TeamAnalyzer:
    """Handles data acquisition, normalization, analytics, and reporting."""

    api_url_template = 'https://api.sportsdata.io/v4/soccer/scores/json/TeamSeasonStats/3/{season}'
    competition_details_template = 'https://api.sportsdata.io/v4/soccer/scores/json/CompetitionDetails/{competition_id}'
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
        source_team = TeamAnalyzer._pick_value(team_stats, 'Name', 'TeamName', 'Team', 'TeamKey')
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

    def load_match_data_from_api(self, api_url, api_key, team_name=default_team_name, log_path=None):
        return _load_match_data_from_api(self, api_url, api_key, team_name, log_path=log_path)

    def load_sample_data(self, api_key=None, seasons=default_seasons, team_name=default_team_name):
        _load_sample_data(self, api_key=api_key, seasons=seasons, team_name=team_name)

    def calculate_basic_stats(self, team_name, season_year=None):
        return _calculate_basic_stats(self.match_data, team_name, season_year=season_year)

    def fetch_competition_details(self, api_key=None, competition_id=3):
        return _fetch_competition_details(self, api_key=api_key, competition_id=competition_id)

    def fetch_all_teams(self, api_key=None, seasons=default_seasons):
        return _fetch_all_teams(self, api_key=api_key, seasons=seasons)

    def _fetch_team_names_for_season(self, season_year, api_key):
        return _fetch_team_names_for_season(self, season_year, api_key)

    def fetch_league_possession_averages(self, api_key=None, seasons=default_seasons):
        return _fetch_league_possession_averages(self, api_key=api_key, seasons=seasons)

    def predict_possession_vs_opponents(self, team_name, opponents, api_key=None, seasons=default_seasons):
        team_possessions = self.fetch_league_possession_averages(api_key=api_key, seasons=seasons)
        return build_predictions(team_name, opponents, team_possessions)

    def generate_report(self, team_name, seasons=default_seasons):
        return _generate_report(self, team_name, seasons)

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
