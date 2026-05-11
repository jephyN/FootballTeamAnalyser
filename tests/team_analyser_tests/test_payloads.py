"""Tests for TeamSeasonStats payload parsing and row extraction."""
# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import numpy as np
import pandas as pd

from team_analyser import TeamAnalyzer, _build_season_row

from .assertions import assert_no_rows, assert_row_count


def _pick(item, *keys):
    """Minimal pick helper matching TeamAnalyzer._pick_value behaviour."""
    for key in keys:
        val = item.get(key)
        if val is not None and val != '':
            return val
    return np.nan


def _season_row(stats):
    """Build a normalized Arsenal row from raw season stats."""
    return _build_season_row(stats, _pick, 'Arsenal FC')


def _normalized_rows(payload):
    """Normalize a raw TeamSeasonStats payload."""
    return TeamAnalyzer._normalize_team_season_payload(payload)


class TestBuildSeasonRow:
    """Tests for the _build_season_row module-level helper."""

    def test_extracts_season_year(self):
        row = _season_row({'Name': 'Arsenal FC', 'Season': 2026, 'Possession': 60.0})
        assert row['season_year'] == 2026

    def test_extracts_possession(self):
        row = _season_row({'Name': 'Arsenal FC', 'Season': 2026, 'Possession': 60.0})
        assert row['possession'] == 60.0

    def test_uses_alternate_key_shots_on_goal(self):
        row = _season_row({'Name': 'Arsenal FC', 'Season': 2026, 'ShotsOnGoal': 75})
        assert row['shots_on_target'] == 75

    def test_date_is_dec_31(self):
        row = _season_row({'Name': 'Arsenal FC', 'Season': 2026})
        assert row['date'] == pd.Timestamp('2026-12-31')

    def test_missing_season_gives_nat(self):
        row = _season_row({'Name': 'Arsenal FC'})
        assert pd.isna(row['date'])

    def test_team_name_stored(self):
        row = _season_row({'Name': 'Arsenal FC', 'Season': 2026})
        assert row['team'] == 'Arsenal FC'


class TestNormalizePayload:
    """Tests for the TeamAnalyzer._normalize_team_season_payload static method."""

    def test_flat_list_passthrough(self, raw_season_stats_payload):
        rows = _normalized_rows(raw_season_stats_payload)
        assert_row_count(rows, 2)
        assert rows[0]['Name'] == 'Arsenal FC'

    def test_wrapped_in_data_key(self, raw_season_stats_payload):
        rows = _normalized_rows({'data': raw_season_stats_payload})
        assert_row_count(rows, 2)

    def test_wrapped_in_team_season_stats_key(self, raw_season_stats_payload):
        rows = _normalized_rows({'TeamSeasonStats': raw_season_stats_payload})
        assert_row_count(rows, 2)

    def test_rounds_structure_flattened(self):
        payload = {
            'Rounds': [{
                'Season': 2026,
                'RoundId': 1,
                'Name': 'Round 1',
                'TeamSeasons': [
                    {'Name': 'Arsenal FC', 'Possession': 60.0},
                    {'Name': 'Chelsea FC', 'Possession': 52.0},
                ],
            }]
        }
        rows = _normalized_rows(payload)
        assert_row_count(rows, 2)
        assert all(row.get('Season') == 2026 for row in rows)

    def test_round_context_backfilled(self):
        payload = {
            'Rounds': [{
                'Season': 2026,
                'SeasonType': 1,
                'RoundId': 5,
                'Name': 'Semi-final',
                'TeamSeasons': [{'Name': 'Arsenal FC'}],
            }]
        }
        rows = _normalized_rows(payload)
        assert rows[0]['Season'] == 2026
        assert rows[0]['RoundId'] == 5

    def test_empty_list_returns_empty(self):
        assert_no_rows(_normalized_rows([]))

    def test_non_dict_non_list_returns_empty(self):
        assert_no_rows(_normalized_rows('invalid'))

    def test_non_dict_items_skipped(self):
        rows = _normalized_rows([None, 42, {'Name': 'Arsenal FC'}])
        assert_row_count(rows, 1)


class TestExtractTeamSeasonRow:
    """Tests for the TeamAnalyzer._extract_team_season_row static method."""

    def test_returns_none_for_wrong_team(self):
        stats = {'Name': 'Chelsea FC', 'Season': 2026}
        assert TeamAnalyzer._extract_team_season_row(stats, 'Arsenal FC') is None

    def test_returns_row_for_correct_team(self):
        stats = {'Name': 'Arsenal FC', 'Season': 2026, 'Possession': 60.0}
        row = TeamAnalyzer._extract_team_season_row(stats, 'Arsenal FC')
        assert row is not None
        assert row['season_year'] == 2026

    def test_alternate_name_key_team_name(self):
        stats = {'TeamName': 'Arsenal FC', 'Season': 2026}
        row = TeamAnalyzer._extract_team_season_row(stats, 'Arsenal FC')
        assert row is not None

    def test_case_insensitive_team_match(self):
        stats = {'Name': 'arsenal fc', 'Season': 2026}
        row = TeamAnalyzer._extract_team_season_row(stats, 'Arsenal FC')
        assert row is not None
