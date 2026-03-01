"""
tests/test_team_analyser.py

Unit tests for team_analyser.py.

Run with:
    pytest tests/

All tests are fully offline — no network calls are made. API methods are
tested by patching TeamAnalyzer._fetch_json so the real urlopen is never
called.
"""

import json
import os
import warnings
from unittest.mock import patch
from urllib.error import URLError

import numpy as np
import pandas as pd
import pytest

from team_analyser import (
    TeamAnalyzer,
    _build_api_headers,
    _coerce_numeric_cols,
    _is_nan,
    _build_season_row,
)

import matplotlib

matplotlib.use('Agg')  # non-interactive backend — no Tk window needed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def analyzer():
    """Return a fresh TeamAnalyzer with no data loaded."""
    return TeamAnalyzer()


@pytest.fixture()
def loaded_analyzer():
    """Return a TeamAnalyzer pre-loaded with two-season fallback data."""
    a = TeamAnalyzer()
    a.match_data = pd.DataFrame({
        'date': pd.to_datetime(['2025-12-31', '2026-12-31']),
        'season_year': [2025, 2026],
        'team': ['Arsenal FC', 'Arsenal FC'],
        'possession': [60.0, 55.0],
        'shots': [200, 180],
        'shots_on_target': [80, 60],
    })
    return a


@pytest.fixture()
def two_team_analyzer():
    """Return a TeamAnalyzer with data for two teams in one season."""
    a = TeamAnalyzer()
    a.match_data = pd.DataFrame({
        'date': pd.to_datetime(['2026-12-31', '2026-12-31']),
        'season_year': [2026, 2026],
        'team': ['Arsenal FC', 'Chelsea FC'],
        'possession': [60.0, 50.0],
        'shots': [200, 150],
        'shots_on_target': [80, 45],
    })
    return a


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


# ---------------------------------------------------------------------------
# _is_nan
# ---------------------------------------------------------------------------

class TestIsNan:
    def test_none_is_nan(self):
        assert _is_nan(None)

    def test_float_nan_is_nan(self):
        assert _is_nan(float('nan'))

    def test_numpy_nan_is_nan(self):
        assert _is_nan(np.nan)

    def test_zero_is_not_nan(self):
        assert not _is_nan(0)

    def test_string_is_not_nan(self):
        assert not _is_nan('N/A')

    def test_valid_float_is_not_nan(self):
        assert not _is_nan(55.5)


# ---------------------------------------------------------------------------
# _build_api_headers
# ---------------------------------------------------------------------------

class TestBuildApiHeaders:
    def test_returns_dict_with_key(self):
        headers = _build_api_headers('my-key')
        assert headers['Ocp-Apim-Subscription-Key'] == 'my-key'

    def test_accept_header_present(self):
        headers = _build_api_headers('x')
        assert headers['Accept'] == 'application/json'


# ---------------------------------------------------------------------------
# _coerce_numeric_cols
# ---------------------------------------------------------------------------

class TestCoerceNumericCols:
    def test_converts_string_numbers(self):
        df = pd.DataFrame({'possession': ['60.5', '55.0'], 'shots': ['200', '150']})
        _coerce_numeric_cols(df)
        assert pd.api.types.is_float_dtype(df['possession'])
        assert pd.api.types.is_numeric_dtype(df['shots'])  # int64 or float64, not object

    def test_ignores_missing_columns(self):
        df = pd.DataFrame({'possession': [60.5]})
        _coerce_numeric_cols(df)  # 'shots' not present — should not raise
        assert df['possession'].dtype == float

    def test_coerces_invalid_to_nan(self):
        df = pd.DataFrame({'possession': ['not-a-number']})
        _coerce_numeric_cols(df)
        assert np.isnan(df['possession'].iloc[0])


# ---------------------------------------------------------------------------
# _build_season_row
# ---------------------------------------------------------------------------

class TestBuildSeasonRow:
    def _pick(self, item, *keys):
        for key in keys:
            val = item.get(key)
            if val is not None and val != '':
                return val
        return np.nan

    def test_extracts_season_year(self):
        stats = {'Name': 'Arsenal FC', 'Season': 2026, 'Possession': 60.0}
        row = _build_season_row(stats, self._pick, 'Arsenal FC')
        assert row['season_year'] == 2026

    def test_extracts_possession(self):
        stats = {'Name': 'Arsenal FC', 'Season': 2026, 'Possession': 60.0}
        row = _build_season_row(stats, self._pick, 'Arsenal FC')
        assert row['possession'] == 60.0

    def test_uses_alternate_key_shots_on_goal(self):
        stats = {'Name': 'Arsenal FC', 'Season': 2026, 'ShotsOnGoal': 75}
        row = _build_season_row(stats, self._pick, 'Arsenal FC')
        assert row['shots_on_target'] == 75

    def test_date_is_dec_31(self):
        stats = {'Name': 'Arsenal FC', 'Season': 2026}
        row = _build_season_row(stats, self._pick, 'Arsenal FC')
        assert row['date'] == pd.Timestamp('2026-12-31')

    def test_missing_season_gives_nat(self):
        stats = {'Name': 'Arsenal FC'}
        row = _build_season_row(stats, self._pick, 'Arsenal FC')
        assert pd.isna(row['date'])

    def test_team_name_stored(self):
        stats = {'Name': 'Arsenal FC', 'Season': 2026}
        row = _build_season_row(stats, self._pick, 'Arsenal FC')
        assert row['team'] == 'Arsenal FC'


# ---------------------------------------------------------------------------
# TeamAnalyzer._fmt
# ---------------------------------------------------------------------------

class TestFmt:
    def test_formats_float(self):
        assert TeamAnalyzer._fmt(56.789) == '56.79'

    def test_nan_returns_na(self):
        assert TeamAnalyzer._fmt(np.nan) == 'N/A'

    def test_none_returns_na(self):
        assert TeamAnalyzer._fmt(None) == 'N/A'

    def test_custom_spec(self):
        assert TeamAnalyzer._fmt(56.789, '.1f') == '56.8'

    def test_integer_value(self):
        assert TeamAnalyzer._fmt(42, '.0f') == '42'


# ---------------------------------------------------------------------------
# TeamAnalyzer._pick_value
# ---------------------------------------------------------------------------

class TestPickValue:
    def test_returns_first_matching_key(self):
        item = {'A': 1, 'B': 2}
        assert TeamAnalyzer._pick_value(item, 'A', 'B') == 1

    def test_skips_none_value(self):
        item = {'A': None, 'B': 2}
        assert TeamAnalyzer._pick_value(item, 'A', 'B') == 2

    def test_skips_empty_string(self):
        item = {'A': '', 'B': 'hello'}
        assert TeamAnalyzer._pick_value(item, 'A', 'B') == 'hello'

    def test_returns_nan_when_no_keys_match(self):
        item = {'C': 3}
        result = TeamAnalyzer._pick_value(item, 'A', 'B')
        assert np.isnan(result)

    def test_zero_is_a_valid_value(self):
        item = {'A': 0}
        assert TeamAnalyzer._pick_value(item, 'A') == 0


# ---------------------------------------------------------------------------
# TeamAnalyzer._team_matches
# ---------------------------------------------------------------------------

class TestTeamMatches:
    def test_exact_match(self):
        assert TeamAnalyzer._team_matches('Arsenal FC', 'Arsenal FC')

    def test_case_insensitive(self):
        assert TeamAnalyzer._team_matches('arsenal fc', 'Arsenal FC')

    def test_whitespace_stripped(self):
        assert TeamAnalyzer._team_matches('  Arsenal FC  ', 'Arsenal FC')

    def test_different_name_returns_false(self):
        assert not TeamAnalyzer._team_matches('Chelsea FC', 'Arsenal FC')

    def test_nan_row_name_returns_false(self):
        assert not TeamAnalyzer._team_matches(np.nan, 'Arsenal FC')


# ---------------------------------------------------------------------------
# TeamAnalyzer._extract_api_key_from_url
# ---------------------------------------------------------------------------

class TestExtractApiKeyFromUrl:
    def test_extracts_key_from_query(self):
        url = 'https://api.example.com/data?key=abc123&season=2026'
        clean, key = TeamAnalyzer._extract_api_key_from_url(url)
        assert key == 'abc123'
        assert 'key=' not in clean
        assert 'season=2026' in clean

    def test_no_key_in_url(self):
        url = 'https://api.example.com/data?season=2026'
        clean, key = TeamAnalyzer._extract_api_key_from_url(url)
        assert key is None
        assert clean == url

    def test_none_url_returns_none_key(self):
        clean, key = TeamAnalyzer._extract_api_key_from_url(None)
        assert clean is None
        assert key is None


# ---------------------------------------------------------------------------
# TeamAnalyzer._normalize_team_season_payload
# ---------------------------------------------------------------------------

class TestNormalizePayload:
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
            'Rounds': [
                {
                    'Season': 2026,
                    'RoundId': 1,
                    'Name': 'Round 1',
                    'TeamSeasons': [
                        {'Name': 'Arsenal FC', 'Possession': 60.0},
                        {'Name': 'Chelsea FC', 'Possession': 52.0},
                    ],
                }
            ]
        }
        rows = TeamAnalyzer._normalize_team_season_payload(payload)
        assert len(rows) == 2
        assert all(r.get('Season') == 2026 for r in rows)

    def test_round_context_backfilled(self):
        payload = {
            'Rounds': [
                {
                    'Season': 2026,
                    'SeasonType': 1,
                    'RoundId': 5,
                    'Name': 'Semi-final',
                    'TeamSeasons': [{'Name': 'Arsenal FC'}],
                }
            ]
        }
        rows = TeamAnalyzer._normalize_team_season_payload(payload)
        assert rows[0]['Season'] == 2026
        assert rows[0]['RoundId'] == 5

    def test_empty_list_returns_empty(self):
        rows = TeamAnalyzer._normalize_team_season_payload([])
        assert rows == []

    def test_non_dict_non_list_returns_empty(self):
        rows = TeamAnalyzer._normalize_team_season_payload('invalid')
        assert rows == []

    def test_non_dict_items_skipped(self):
        rows = TeamAnalyzer._normalize_team_season_payload([None, 42, {'Name': 'Arsenal FC'}])
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# TeamAnalyzer._extract_team_season_row
# ---------------------------------------------------------------------------

class TestExtractTeamSeasonRow:
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


# ---------------------------------------------------------------------------
# TeamAnalyzer._default_match_data
# ---------------------------------------------------------------------------

class TestDefaultMatchData:
    def test_returns_dataframe(self):
        df = TeamAnalyzer._default_match_data()
        assert isinstance(df, pd.DataFrame)

    def test_has_two_rows(self):
        df = TeamAnalyzer._default_match_data()
        assert len(df) == 2

    def test_custom_team_name(self):
        df = TeamAnalyzer._default_match_data('Chelsea FC')
        assert (df['team'] == 'Chelsea FC').all()

    def test_seasons_present(self):
        df = TeamAnalyzer._default_match_data()
        assert set(df['season_year']) == {2025, 2026}


# ---------------------------------------------------------------------------
# TeamAnalyzer.calculate_basic_stats
# ---------------------------------------------------------------------------

class TestCalculateBasicStats:
    def test_avg_possession_correct(self, loaded_analyzer):
        stats = loaded_analyzer.calculate_basic_stats('Arsenal FC')
        assert stats['avg_possession'] == pytest.approx(57.5)

    def test_shot_accuracy_correct(self, loaded_analyzer):
        # 2025: 80/200 = 40%; 2026: 60/180 = 33.33...%
        # mean = (40 + 33.33) / 2 = 36.67%
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


# ---------------------------------------------------------------------------
# TeamAnalyzer.generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:
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


# ---------------------------------------------------------------------------
# TeamAnalyzer.load_sample_data — fallback path
# ---------------------------------------------------------------------------

class TestLoadSampleDataFallback:
    def test_uses_fallback_when_no_api_key(self, analyzer):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('SPORTSDATA_API_KEY', None)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                analyzer.load_sample_data(api_key=None)
        assert analyzer.match_data is not None
        assert len(analyzer.match_data) == 2

    def test_fallback_has_correct_team(self, analyzer):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('SPORTSDATA_API_KEY', None)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                analyzer.load_sample_data(api_key=None, team_name='Chelsea FC')
        assert (analyzer.match_data['team'] == 'Chelsea FC').all()

    def test_warns_when_no_key(self, analyzer):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(TeamAnalyzer, '_load_env_file'):  # prevent .env loading
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    analyzer.load_sample_data(api_key=None)
        messages = [str(w.message) for w in caught]
        assert any('SPORTSDATA_API_KEY' in m for m in messages)


# ---------------------------------------------------------------------------
# TeamAnalyzer.load_sample_data — API path (mocked)
# ---------------------------------------------------------------------------

class TestLoadSampleDataApi:
    def _make_payload(self, team_name, season):
        return [
            {
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
            }
        ]

    def test_loads_two_seasons(self, analyzer):
        payloads = [
            self._make_payload('Arsenal FC', 2025),
            self._make_payload('Arsenal FC', 2026),
        ]
        with patch.object(TeamAnalyzer, '_fetch_json', side_effect=payloads):
            analyzer.load_sample_data(api_key='fake-key', team_name='Arsenal FC')
        assert set(analyzer.match_data['season_year']) == {2025, 2026}

    def test_skips_failed_season_with_warning(self, analyzer):
        def fail_first(url, **_):
            if '2025' in url:
                raise ValueError('API error')
            return self._make_payload('Arsenal FC', 2026)

        with patch.object(TeamAnalyzer, '_fetch_json', side_effect=fail_first):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                analyzer.load_sample_data(
                    api_key='fake-key', team_name='Arsenal FC', seasons=(2025, 2026)
                )
        assert len(analyzer.match_data) == 1
        assert any('2025' in str(w.message) for w in caught)

    def test_falls_back_when_all_seasons_fail(self, analyzer):
        with patch.object(TeamAnalyzer, '_fetch_json', side_effect=URLError('timeout')):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                analyzer.load_sample_data(api_key='fake-key', seasons=(2025, 2026))
        assert analyzer.match_data is not None  # fallback data loaded

    def test_data_sorted_by_date(self, analyzer):
        # Return 2026 first, 2025 second — result should be sorted ascending
        payloads = [
            self._make_payload('Arsenal FC', 2026),
            self._make_payload('Arsenal FC', 2025),
        ]
        with patch.object(TeamAnalyzer, '_fetch_json', side_effect=payloads):
            analyzer.load_sample_data(
                api_key='fake-key', team_name='Arsenal FC', seasons=(2026, 2025)
            )
        years = list(analyzer.match_data['season_year'])
        assert years == sorted(years)


# ---------------------------------------------------------------------------
# TeamAnalyzer.load_match_data_from_api
# ---------------------------------------------------------------------------

class TestLoadMatchDataFromApi:
    def test_raises_without_api_key(self, analyzer):
        with pytest.raises(ValueError, match='API key'):
            analyzer.load_match_data_from_api(
                api_url='https://example.com', api_key=None
            )

    def test_raises_when_team_not_in_payload(self, analyzer, raw_season_stats_payload):
        with patch.object(TeamAnalyzer, '_fetch_json', return_value=raw_season_stats_payload):
            with pytest.raises(ValueError, match='Unknown FC'):
                analyzer.load_match_data_from_api(
                    api_url='https://example.com',
                    api_key='key',
                    team_name='Unknown FC',
                )

    def test_returns_dataframe_for_valid_team(self, analyzer, raw_season_stats_payload):
        with patch.object(TeamAnalyzer, '_fetch_json', return_value=raw_season_stats_payload):
            df = analyzer.load_match_data_from_api(
                api_url='https://example.com',
                api_key='key',
                team_name='Arsenal FC',
            )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]['season_year'] == 2026

    def test_numeric_cols_are_float(self, analyzer, raw_season_stats_payload):
        with patch.object(TeamAnalyzer, '_fetch_json', return_value=raw_season_stats_payload):
            df = analyzer.load_match_data_from_api(
                api_url='https://example.com', api_key='key', team_name='Arsenal FC'
            )
        assert pd.api.types.is_float_dtype(df['possession'])
        assert pd.api.types.is_numeric_dtype(df['shots'])  # int64 or float64, not object

    def test_extracts_key_from_url(self, analyzer, raw_season_stats_payload):
        """Key embedded in URL should be accepted even with api_key=None."""
        url = 'https://example.com?key=embedded-key'
        with patch.object(TeamAnalyzer, '_fetch_json', return_value=raw_season_stats_payload):
            df = analyzer.load_match_data_from_api(
                api_url=url, api_key=None, team_name='Arsenal FC'
            )
        assert df is not None


# ---------------------------------------------------------------------------
# TeamAnalyzer.fetch_competition_details
# ---------------------------------------------------------------------------

class TestFetchCompetitionDetails:
    def test_returns_empty_without_key(self, analyzer):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('SPORTSDATA_API_KEY', None)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                result = analyzer.fetch_competition_details(api_key=None)
        assert result == {}

    def test_parses_teams_key(self, analyzer):
        payload = {
            'Teams': [
                {'Name': 'Arsenal FC', 'WikipediaLogoUrl': 'https://upload.wikimedia.org/...'},
                {'Name': 'Chelsea FC', 'WikipediaLogoUrl': None},
            ]
        }
        with patch.object(TeamAnalyzer, '_fetch_json', return_value=payload):
            result = analyzer.fetch_competition_details(api_key='key')
        assert 'Arsenal FC' in result
        assert 'Chelsea FC' in result
        assert result['Arsenal FC'] == 'https://upload.wikimedia.org/...'
        assert result['Chelsea FC'] is None

    def test_returns_empty_on_network_error(self, analyzer):
        with patch.object(TeamAnalyzer, '_fetch_json', side_effect=URLError('timeout')):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                result = analyzer.fetch_competition_details(api_key='key')
        assert result == {}

    def test_falls_back_to_team_name_key(self, analyzer):
        payload = {'Teams': [{'TeamName': 'Juventus', 'WikipediaLogoUrl': None}]}
        with patch.object(TeamAnalyzer, '_fetch_json', return_value=payload):
            result = analyzer.fetch_competition_details(api_key='key')
        assert 'Juventus' in result


# ---------------------------------------------------------------------------
# TeamAnalyzer.fetch_all_teams
# ---------------------------------------------------------------------------

class TestFetchAllTeams:
    def test_returns_fallback_without_key(self, analyzer):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('SPORTSDATA_API_KEY', None)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                teams = analyzer.fetch_all_teams(api_key=None)
        assert teams == [TeamAnalyzer.default_team_name]

    def test_returns_sorted_deduplicated_list(self, analyzer, raw_season_stats_payload):
        with patch.object(TeamAnalyzer, '_fetch_json', return_value=raw_season_stats_payload):
            teams = analyzer.fetch_all_teams(api_key='key', seasons=(2026,))
        assert teams == sorted(teams)
        assert len(teams) == len(set(teams))
        assert 'Arsenal FC' in teams
        assert 'Chelsea FC' in teams

    def test_deduplicates_across_seasons(self, analyzer, raw_season_stats_payload):
        # Both seasons return the same teams — result should not double-count.
        with patch.object(
                TeamAnalyzer, '_fetch_json', return_value=raw_season_stats_payload
        ):
            teams = analyzer.fetch_all_teams(api_key='key', seasons=(2025, 2026))
        assert teams.count('Arsenal FC') == 1

    def test_skips_failed_season(self, analyzer, raw_season_stats_payload):
        def fail_first(url, **_):
            if '2025' in url:
                raise URLError('timeout')
            return raw_season_stats_payload

        with patch.object(TeamAnalyzer, '_fetch_json', side_effect=fail_first):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                teams = analyzer.fetch_all_teams(api_key='key', seasons=(2025, 2026))
        assert 'Arsenal FC' in teams  # still present from 2026

    def test_returns_fallback_when_all_seasons_fail(self, analyzer):
        with patch.object(TeamAnalyzer, '_fetch_json', side_effect=URLError('timeout')):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                teams = analyzer.fetch_all_teams(api_key='key', seasons=(2026,))
        assert teams == [TeamAnalyzer.default_team_name]


# ---------------------------------------------------------------------------
# TeamAnalyzer.plot_performance_trends
# ---------------------------------------------------------------------------

class TestPlotPerformanceTrends:
    def test_returns_figure(self, loaded_analyzer):
        import matplotlib.pyplot as plt
        fig = loaded_analyzer.plot_performance_trends('Arsenal FC', seasons=(2025, 2026))
        assert fig is not None
        plt.close(fig)

    def test_figure_has_two_axes(self, loaded_analyzer):
        import matplotlib.pyplot as plt
        fig = loaded_analyzer.plot_performance_trends('Arsenal FC', seasons=(2025, 2026))
        # tight_layout may add a third logo axes — filter to named subplots only
        subplots = [ax for ax in fig.axes if ax.get_subplotspec() is not None]
        assert len(subplots) == 2
        plt.close(fig)

    def test_suptitle_contains_team_name(self, loaded_analyzer):
        import matplotlib.pyplot as plt
        fig = loaded_analyzer.plot_performance_trends('Arsenal FC')
        assert 'Arsenal FC' in fig.texts[0].get_text()
        plt.close(fig)

    def test_logo_axes_added_when_logo_present(self, loaded_analyzer):
        import matplotlib.pyplot as plt
        from PIL import Image
        fake_logo = Image.new('RGBA', (52, 52), color=(255, 0, 0, 255))
        with patch('team_analyser._fetch_logo_pil', return_value=fake_logo):
            fig = loaded_analyzer.plot_performance_trends(
                'Arsenal FC', logo_url='https://example.com/logo.png'
            )
        # With logo there should be 3 axes (2 subplots + 1 logo)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_no_logo_axes_without_logo(self, loaded_analyzer):
        import matplotlib.pyplot as plt
        with patch('team_analyser._fetch_logo_pil', return_value=None):
            fig = loaded_analyzer.plot_performance_trends(
                'Arsenal FC', logo_url='https://example.com/logo.png'
            )
        assert len(fig.axes) == 2
        plt.close(fig)


# ---------------------------------------------------------------------------
# TeamAnalyzer._log_raw_api_data
# ---------------------------------------------------------------------------

class TestLogRawApiData:
    def test_creates_json_file(self, tmp_path):
        TeamAnalyzer._log_buffer = {}
        log_file = tmp_path / 'test_log.json'
        TeamAnalyzer._log_raw_api_data(
            2026, [{'Name': 'Arsenal FC'}], 'Arsenal FC', log_path=log_file
        )
        assert log_file.exists()

    def test_json_contains_season_data(self, tmp_path):
        TeamAnalyzer._log_buffer = {}
        log_file = tmp_path / 'test_log.json'
        TeamAnalyzer._log_raw_api_data(
            2026, [{'Name': 'Arsenal FC'}], 'Arsenal FC', log_path=log_file
        )
        data = json.loads(log_file.read_text())
        assert '2026' in data['api_log']
        assert data['api_log']['2026']['row_count'] == 1

    def test_accumulates_multiple_seasons(self, tmp_path):
        TeamAnalyzer._log_buffer = {}
        log_file = tmp_path / 'test_log.json'
        TeamAnalyzer._log_raw_api_data(2025, [], 'Arsenal FC', log_path=log_file)
        TeamAnalyzer._log_raw_api_data(2026, [], 'Arsenal FC', log_path=log_file)
        data = json.loads(log_file.read_text())
        assert '2025' in data['api_log']
        assert '2026' in data['api_log']


# ---------------------------------------------------------------------------
# TeamAnalyzer._load_env_file
# ---------------------------------------------------------------------------

class TestLoadEnvFile:
    def test_loads_key_from_file(self, tmp_path, monkeypatch):
        env_file = tmp_path / '.env'
        env_file.write_text('SPORTSDATA_API_KEY=test-key-123\n')
        monkeypatch.delenv('SPORTSDATA_API_KEY', raising=False)
        TeamAnalyzer._load_env_file(str(env_file))
        assert os.environ.get('SPORTSDATA_API_KEY') == 'test-key-123'

    def test_strips_quotes(self, tmp_path, monkeypatch):
        env_file = tmp_path / '.env'
        env_file.write_text('SPORTSDATA_API_KEY="quoted-key"\n')
        monkeypatch.delenv('SPORTSDATA_API_KEY', raising=False)
        TeamAnalyzer._load_env_file(str(env_file))
        assert os.environ['SPORTSDATA_API_KEY'] == 'quoted-key'

    def test_ignores_comments(self, tmp_path, monkeypatch):
        env_file = tmp_path / '.env'
        env_file.write_text('# comment\nSPORTSDATA_API_KEY=real-key\n')
        monkeypatch.delenv('SPORTSDATA_API_KEY', raising=False)
        TeamAnalyzer._load_env_file(str(env_file))
        assert os.environ['SPORTSDATA_API_KEY'] == 'real-key'

    def test_does_not_override_existing_env(self, tmp_path, monkeypatch):
        env_file = tmp_path / '.env'
        env_file.write_text('SPORTSDATA_API_KEY=from-file\n')
        monkeypatch.setenv('SPORTSDATA_API_KEY', 'already-set')
        TeamAnalyzer._load_env_file(str(env_file))
        assert os.environ['SPORTSDATA_API_KEY'] == 'already-set'

    def test_missing_file_does_not_raise(self):
        TeamAnalyzer._load_env_file('/nonexistent/.env')  # should silently return
