"""
conftest.py

Adds the project root to sys.path so that test modules can import
team_analyser, logo_utils, gui, and main without installation.
"""

import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Insert the directory containing team_analyser.py, gui.py etc.
sys.path.insert(0, str(Path(__file__).parent.parent))

from team_analyser import TeamAnalyzer  # pylint: disable=wrong-import-position


@pytest.fixture()
def analyzer():
    """Return a fresh TeamAnalyzer with no data loaded."""
    return TeamAnalyzer()


@pytest.fixture()
def loaded_analyzer():
    """Return a TeamAnalyzer pre-loaded with two-season fallback data."""
    analyser = TeamAnalyzer()
    analyser.match_data = pd.DataFrame({
        'date': pd.to_datetime(['2025-12-31', '2026-12-31']),
        'season_year': [2025, 2026],
        'team': ['Arsenal FC', 'Arsenal FC'],
        'possession': [60.0, 55.0],
        'shots': [200, 180],
        'shots_on_target': [80, 60],
    })
    return analyser


@pytest.fixture()
def two_team_analyzer():
    """Return a TeamAnalyzer with data for two teams in one season."""
    analyser = TeamAnalyzer()
    analyser.match_data = pd.DataFrame({
        'date': pd.to_datetime(['2026-12-31', '2026-12-31']),
        'season_year': [2026, 2026],
        'team': ['Arsenal FC', 'Chelsea FC'],
        'possession': [60.0, 50.0],
        'shots': [200, 150],
        'shots_on_target': [80, 45],
    })
    return analyser


@pytest.fixture()
def make_season_payload():
    """Return a factory for minimal TeamSeasonStats API payloads."""
    def _make_payload(team_name='Arsenal FC', season=2026):
        return [{
            'Name': team_name,
            'Season': season,
            'Possession': 60.0,
            'Shots': 200,
            'ShotsOnGoal': 80,
            'Score': 40,
            'OpponentScore': 15,
            'Games': 8,
            'GoalkeeperCleanSheets': 3,
            'SeasonType': 1,
            'TeamId': 1,
        }]

    return _make_payload


@pytest.fixture()
def raw_season_stats_payload():
    """Minimal flat-list payload as returned by the TeamSeasonStats API."""
    return [
        {
            'Name': 'Arsenal FC',
            'Season': 2026,
            'Possession': 60.5,
            'Shots': 200,
            'ShotsOnGoal': 80,
            'Score': 45,
            'OpponentScore': 20,
            'Games': 8,
            'GoalkeeperCleanSheets': 3,
            'SeasonType': 1,
            'TeamId': 42,
        },
        {
            'Name': 'Chelsea FC',
            'Season': 2026,
            'Possession': 52.0,
            'Shots': 160,
            'ShotsOnGoal': 55,
            'Score': 30,
            'OpponentScore': 25,
            'Games': 8,
            'GoalkeeperCleanSheets': 2,
            'SeasonType': 1,
            'TeamId': 7,
        },
    ]


@pytest.fixture()
def patched_fetch_json():
    """Return a context manager that patches TeamAnalyzer._fetch_json."""
    @contextmanager
    def _patched_fetch_json(return_value=None, side_effect=None):
        with patch.object(
            TeamAnalyzer, '_fetch_json', return_value=return_value, side_effect=side_effect
        ):
            yield

    return _patched_fetch_json


@pytest.fixture()
def without_api_key():
    """Return a context manager that clears API-key state and captures warnings."""
    @contextmanager
    def _without_api_key():
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('SPORTSDATA_API_KEY', None)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                yield caught

    return _without_api_key
