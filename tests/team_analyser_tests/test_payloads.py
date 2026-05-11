"""Tests for TeamSeasonStats payload parsing and row extraction."""

import numpy as np
import pandas as pd

from team_analyser import TeamAnalyzer, _build_season_row


def _pick(item, *keys):
    """Minimal pick helper matching TeamAnalyzer._pick_value behaviour."""
    for key in keys:
        val = item.get(key)
        if val is not None and val != '':
            return val
    return np.nan


class TestBuildSeasonRow:
    """Tests for the _build_season_row module-level helper."""

    def test_extracts_season_year(self):
        stats = {'Name': 'Arsenal FC', 'Season': 2026, 'Possession': 60.0}
        row = _build_season_row(stats, _pick, 'Arsenal FC')
        assert row['season_year'] == 2026

    def test_extracts_possession(self):
        stats = {'Name': 'Arsenal FC', 'Season': 2026, 'Possession': 60.0}
        row = _build_season_row(stats, _pick, 'Arsenal FC')
        assert row['possession'] == 60.0

    def test_uses_alternate_key_shots_on_goal(self):
        stats = {'Name': 'Arsenal FC', 'Season': 2026, 'ShotsOnGoal': 75}
        row = _build_season_row(stats, _pick, 'Arsenal FC')
        assert row['shots_on_target'] == 75

    def test_date_is_dec_31(self):
        stats = {'Name': 'Arsenal FC', 'Season': 2026}
        row = _build_season_row(stats, _pick, 'Arsenal FC')
        assert row['date'] == pd.Timestamp('2026-12-31')

    def test_missing_season_gives_nat(self):
        stats = {'Name': 'Arsenal FC'}
        row = _build_season_row(stats, _pick, 'Arsenal FC')
        assert pd.isna(row['date'])

    def test_team_name_stored(self):
        stats = {'Name': 'Arsenal FC', 'Season': 2026}
        row = _build_season_row(stats, _pick, 'Arsenal FC')
        assert row['team'] == 'Arsenal FC'


class TestNormalizePayload:
    """Tests for the TeamAnalyzer._normalize_team_season_payload static method."""

    def test_flat_list_passthrough(self, raw_season_stats_payload):
        rows = TeamAnalyzer._normalize_team_season_payload(raw_season_stats_payload)
        assert len(rows) == 2
        assert rows[0]['Name'] == 'Arsenal FC'

    def test_wrapped_in_data_key(self, raw_season_stats_payload):
        payload = {'data': raw_season_stats_payload}
        rows = TeamAnalyzer._normalize_team_season_payload(payload)
        assert len(rows) == 2

    def test_wrapped_in_team_season_stats_key(self, raw_season_stats_payload):
        payload = {'TeamSeasonStats': raw_season_stats_payload}
        rows = TeamAnalyzer._normalize_team_season_payload(payload)
        assert len(rows) == 2

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
        rows = TeamAnalyzer._normalize_team_season_payload(payload)
        assert len(rows) == 2
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
        rows = TeamAnalyzer._normalize_team_season_payload(payload)
        assert rows[0]['Season'] == 2026
        assert rows[0]['RoundId'] == 5

    def test_empty_list_returns_empty(self):
        rows = TeamAnalyzer._normalize_team_season_payload([])
        assert not rows

    def test_non_dict_non_list_returns_empty(self):
        rows = TeamAnalyzer._normalize_team_season_payload('invalid')
        assert not rows

    def test_non_dict_items_skipped(self):
        rows = TeamAnalyzer._normalize_team_season_payload(
            [None, 42, {'Name': 'Arsenal FC'}]
        )
        assert len(rows) == 1


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
