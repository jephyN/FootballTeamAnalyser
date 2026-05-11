"""Tests for calculated metrics and text reports."""
# pylint: disable=missing-function-docstring,redefined-outer-name

import pandas as pd
import pytest

from team_analyser import _is_nan


class TestCalculateBasicStats:
    """Tests for TeamAnalyzer.calculate_basic_stats."""

    def test_avg_possession_correct(self, loaded_analyzer):
        stats = loaded_analyzer.calculate_basic_stats('Arsenal FC')
        assert stats['avg_possession'] == pytest.approx(57.5)

    def test_shot_accuracy_correct(self, loaded_analyzer):
        stats = loaded_analyzer.calculate_basic_stats('Arsenal FC')
        assert stats['shot_accuracy'] == pytest.approx(36.67, abs=0.01)

    def test_filtered_by_season(self, loaded_analyzer):
        stats = loaded_analyzer.calculate_basic_stats('Arsenal FC', season_year=2025)
        assert stats['avg_possession'] == pytest.approx(60.0)

    def test_returns_nan_for_unknown_team(self, loaded_analyzer):
        stats = loaded_analyzer.calculate_basic_stats('Unknown FC')
        assert _is_nan(stats['avg_possession'])
        assert _is_nan(stats['shot_accuracy'])

    def test_returns_nan_when_no_shots(self, analyzer):
        analyzer.match_data = pd.DataFrame({
            'season_year': [2026],
            'team': ['Arsenal FC'],
            'possession': [60.0],
            'shots': [0],
            'shots_on_target': [0],
        })
        stats = analyzer.calculate_basic_stats('Arsenal FC')
        assert _is_nan(stats['shot_accuracy'])

    def test_returns_nan_when_possession_missing(self, analyzer):
        analyzer.match_data = pd.DataFrame({
            'season_year': [2026],
            'team': ['Arsenal FC'],
            'shots': [100],
            'shots_on_target': [40],
        })
        stats = analyzer.calculate_basic_stats('Arsenal FC')
        assert _is_nan(stats['avg_possession'])

    def test_isolates_team_from_multi_team_data(self, two_team_analyzer):
        arsenal = two_team_analyzer.calculate_basic_stats('Arsenal FC')
        chelsea = two_team_analyzer.calculate_basic_stats('Chelsea FC')
        assert arsenal['avg_possession'] != chelsea['avg_possession']
        assert arsenal['avg_possession'] == pytest.approx(60.0)
        assert chelsea['avg_possession'] == pytest.approx(50.0)


class TestGenerateReport:
    """Tests for TeamAnalyzer.generate_report."""

    def test_contains_team_name(self, loaded_analyzer):
        report = loaded_analyzer.generate_report('Arsenal FC', seasons=(2025, 2026))
        assert 'Arsenal FC' in report

    def test_contains_both_seasons(self, loaded_analyzer):
        report = loaded_analyzer.generate_report('Arsenal FC', seasons=(2025, 2026))
        assert '2025' in report
        assert '2026' in report

    def test_contains_possession_label(self, loaded_analyzer):
        report = loaded_analyzer.generate_report('Arsenal FC', seasons=(2025, 2026))
        assert 'Avg Possession' in report

    def test_contains_shot_accuracy_label(self, loaded_analyzer):
        report = loaded_analyzer.generate_report('Arsenal FC', seasons=(2025, 2026))
        assert 'Shot Accuracy' in report

    def test_no_data_message_when_season_absent(self, loaded_analyzer):
        report = loaded_analyzer.generate_report('Arsenal FC', seasons=(2099,))
        assert 'No data available' in report

    def test_percentage_symbol_present(self, loaded_analyzer):
        report = loaded_analyzer.generate_report('Arsenal FC', seasons=(2025, 2026))
        assert '%' in report
