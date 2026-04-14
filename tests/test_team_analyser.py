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
import builtins
from unittest.mock import patch
from urllib.error import URLError

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from team_analyser import (
    TeamAnalyzer,
    _build_api_headers,
    _build_possession_training_frame,
    _build_season_row,
    _coerce_numeric_cols,
    _is_nan,
)

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
    """Tests for the _is_nan module-level helper."""

    def test_none_is_nan(self):
        """None is treated as NaN."""
        assert _is_nan(None)

    def test_float_nan_is_nan(self):
        """float('nan') is treated as NaN."""
        assert _is_nan(float('nan'))

    def test_numpy_nan_is_nan(self):
        """numpy.nan is treated as NaN."""
        assert _is_nan(np.nan)

    def test_zero_is_not_nan(self):
        """Zero is a valid value, not NaN."""
        assert not _is_nan(0)

    def test_string_is_not_nan(self):
        """A non-empty string is not NaN."""
        assert not _is_nan('N/A')

    def test_valid_float_is_not_nan(self):
        """A regular float is not NaN."""
        assert not _is_nan(55.5)


# ---------------------------------------------------------------------------
# _build_api_headers
# ---------------------------------------------------------------------------

class TestBuildApiHeaders:
    """Tests for the _build_api_headers module-level helper."""

    def test_returns_dict_with_key(self):
        """The subscription key is placed in the correct header."""
        headers = _build_api_headers('my-key')
        assert headers['Ocp-Apim-Subscription-Key'] == 'my-key'

    def test_accept_header_present(self):
        """The Accept: application/json header is always included."""
        headers = _build_api_headers('x')
        assert headers['Accept'] == 'application/json'


# ---------------------------------------------------------------------------
# _coerce_numeric_cols
# ---------------------------------------------------------------------------

class TestCoerceNumericCols:
    """Tests for the _coerce_numeric_cols module-level helper."""

    def test_converts_string_numbers(self):
        """String-encoded numbers are coerced to a numeric dtype."""
        df = pd.DataFrame({'possession': ['60.5', '55.0'], 'shots': ['200', '150']})
        _coerce_numeric_cols(df)
        assert pd.api.types.is_float_dtype(df['possession'])
        assert pd.api.types.is_numeric_dtype(df['shots'])

    def test_ignores_missing_columns(self):
        """Columns not in _NUMERIC_COLS are silently ignored."""
        df = pd.DataFrame({'possession': [60.5]})
        _coerce_numeric_cols(df)
        assert df['possession'].dtype == float

    def test_coerces_invalid_to_nan(self):
        """Non-numeric strings are coerced to NaN rather than raising."""
        df = pd.DataFrame({'possession': ['not-a-number']})
        _coerce_numeric_cols(df)
        assert np.isnan(df['possession'].iloc[0])


# ---------------------------------------------------------------------------
# _build_season_row
# ---------------------------------------------------------------------------

class TestBuildSeasonRow:
    """Tests for the _build_season_row module-level helper."""

    def _pick(self, item, *keys):
        """Minimal pick helper matching TeamAnalyzer._pick_value behaviour."""
        for key in keys:
            val = item.get(key)
            if val is not None and val != '':
                return val
        return np.nan

    def test_extracts_season_year(self):
        """Season year is correctly extracted from the Season key."""
        stats = {'Name': 'Arsenal FC', 'Season': 2026, 'Possession': 60.0}
        row = _build_season_row(stats, self._pick, 'Arsenal FC')
        assert row['season_year'] == 2026

    def test_extracts_possession(self):
        """Possession value is forwarded into the row dict."""
        stats = {'Name': 'Arsenal FC', 'Season': 2026, 'Possession': 60.0}
        row = _build_season_row(stats, self._pick, 'Arsenal FC')
        assert row['possession'] == 60.0

    def test_uses_alternate_key_shots_on_goal(self):
        """ShotsOnGoal is accepted as an alternate key for shots_on_target."""
        stats = {'Name': 'Arsenal FC', 'Season': 2026, 'ShotsOnGoal': 75}
        row = _build_season_row(stats, self._pick, 'Arsenal FC')
        assert row['shots_on_target'] == 75

    def test_date_is_dec_31(self):
        """The row date is set to 31 December of the season year."""
        stats = {'Name': 'Arsenal FC', 'Season': 2026}
        row = _build_season_row(stats, self._pick, 'Arsenal FC')
        assert row['date'] == pd.Timestamp('2026-12-31')

    def test_missing_season_gives_nat(self):
        """A missing Season key produces NaT for the date field."""
        stats = {'Name': 'Arsenal FC'}
        row = _build_season_row(stats, self._pick, 'Arsenal FC')
        assert pd.isna(row['date'])

    def test_team_name_stored(self):
        """The provided team name is stored in the row dict."""
        stats = {'Name': 'Arsenal FC', 'Season': 2026}
        row = _build_season_row(stats, self._pick, 'Arsenal FC')
        assert row['team'] == 'Arsenal FC'


# ---------------------------------------------------------------------------
# TeamAnalyzer._fmt
# ---------------------------------------------------------------------------

# pylint: disable=protected-access
class TestFmt:
    """Tests for the TeamAnalyzer._fmt static formatting helper."""

    def test_formats_float(self):
        """A regular float is formatted to two decimal places by default."""
        assert TeamAnalyzer._fmt(56.789) == '56.79'

    def test_nan_returns_na(self):
        """NaN produces the sentinel string 'N/A'."""
        assert TeamAnalyzer._fmt(np.nan) == 'N/A'

    def test_none_returns_na(self):
        """None produces the sentinel string 'N/A'."""
        assert TeamAnalyzer._fmt(None) == 'N/A'

    def test_custom_spec(self):
        """A custom format spec is applied correctly."""
        assert TeamAnalyzer._fmt(56.789, '.1f') == '56.8'

    def test_integer_value(self):
        """Integer values are formatted without raising."""
        assert TeamAnalyzer._fmt(42, '.0f') == '42'
# pylint: enable=protected-access


# ---------------------------------------------------------------------------
# TeamAnalyzer._pick_value
# ---------------------------------------------------------------------------

# pylint: disable=protected-access
class TestPickValue:
    """Tests for the TeamAnalyzer._pick_value static helper."""

    def test_returns_first_matching_key(self):
        """The value of the first present key is returned."""
        item = {'A': 1, 'B': 2}
        assert TeamAnalyzer._pick_value(item, 'A', 'B') == 1

    def test_skips_none_value(self):
        """A None value is skipped in favour of the next candidate key."""
        item = {'A': None, 'B': 2}
        assert TeamAnalyzer._pick_value(item, 'A', 'B') == 2

    def test_skips_empty_string(self):
        """An empty string is skipped in favour of the next candidate key."""
        item = {'A': '', 'B': 'hello'}
        assert TeamAnalyzer._pick_value(item, 'A', 'B') == 'hello'

    def test_returns_nan_when_no_keys_match(self):
        """numpy.nan is returned when none of the candidate keys are present."""
        item = {'C': 3}
        result = TeamAnalyzer._pick_value(item, 'A', 'B')
        assert np.isnan(result)

    def test_zero_is_a_valid_value(self):
        """Zero is a legitimate value and must not be skipped."""
        item = {'A': 0}
        assert TeamAnalyzer._pick_value(item, 'A') == 0
# pylint: enable=protected-access


# ---------------------------------------------------------------------------
# TeamAnalyzer._team_matches
# ---------------------------------------------------------------------------

# pylint: disable=protected-access
class TestTeamMatches:
    """Tests for the TeamAnalyzer._team_matches static helper."""

    def test_exact_match(self):
        """Identical team names match."""
        assert TeamAnalyzer._team_matches('Arsenal FC', 'Arsenal FC')

    def test_case_insensitive(self):
        """Comparison is case-insensitive."""
        assert TeamAnalyzer._team_matches('arsenal fc', 'Arsenal FC')

    def test_whitespace_stripped(self):
        """Leading and trailing whitespace is stripped before comparison."""
        assert TeamAnalyzer._team_matches('  Arsenal FC  ', 'Arsenal FC')

    def test_different_name_returns_false(self):
        """Different team names do not match."""
        assert not TeamAnalyzer._team_matches('Chelsea FC', 'Arsenal FC')

    def test_nan_row_name_returns_false(self):
        """A NaN row team name never matches."""
        assert not TeamAnalyzer._team_matches(np.nan, 'Arsenal FC')
# pylint: enable=protected-access


# ---------------------------------------------------------------------------
# TeamAnalyzer._extract_api_key_from_url
# ---------------------------------------------------------------------------

# pylint: disable=protected-access
class TestExtractApiKeyFromUrl:
    """Tests for the TeamAnalyzer._extract_api_key_from_url static helper."""

    def test_extracts_key_from_query(self):
        """The key query param is extracted and removed from the URL."""
        url = 'https://api.example.com/data?key=abc123&season=2026'
        clean, key = TeamAnalyzer._extract_api_key_from_url(url)
        assert key == 'abc123'
        assert 'key=' not in clean
        assert 'season=2026' in clean

    def test_no_key_in_url(self):
        """URLs without a key param return None for the key."""
        url = 'https://api.example.com/data?season=2026'
        clean, key = TeamAnalyzer._extract_api_key_from_url(url)
        assert key is None
        assert clean == url

    def test_none_url_returns_none_key(self):
        """A None URL is handled gracefully."""
        clean, key = TeamAnalyzer._extract_api_key_from_url(None)
        assert clean is None
        assert key is None
# pylint: enable=protected-access


# ---------------------------------------------------------------------------
# TeamAnalyzer._normalize_team_season_payload
# ---------------------------------------------------------------------------

# pylint: disable=protected-access,redefined-outer-name
class TestNormalizePayload:
    """Tests for the TeamAnalyzer._normalize_team_season_payload static method."""

    def test_flat_list_passthrough(self, raw_season_stats_payload):
        """A plain list of dicts is returned as-is."""
        rows = TeamAnalyzer._normalize_team_season_payload(raw_season_stats_payload)
        assert len(rows) == 2
        assert rows[0]['Name'] == 'Arsenal FC'

    def test_wrapped_in_data_key(self, raw_season_stats_payload):
        """Payload wrapped under a 'data' key is unwrapped correctly."""
        payload = {'data': raw_season_stats_payload}
        rows = TeamAnalyzer._normalize_team_season_payload(payload)
        assert len(rows) == 2

    def test_wrapped_in_team_season_stats_key(self, raw_season_stats_payload):
        """Payload wrapped under 'TeamSeasonStats' is unwrapped correctly."""
        payload = {'TeamSeasonStats': raw_season_stats_payload}
        rows = TeamAnalyzer._normalize_team_season_payload(payload)
        assert len(rows) == 2

    def test_rounds_structure_flattened(self):
        """Nested Rounds/TeamSeasons structure is flattened to a row per team."""
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
        """Round-level Season and RoundId are backfilled into child rows."""
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
        """An empty list payload produces an empty result."""
        rows = TeamAnalyzer._normalize_team_season_payload([])
        assert not rows

    def test_non_dict_non_list_returns_empty(self):
        """A non-list, non-dict payload produces an empty result."""
        rows = TeamAnalyzer._normalize_team_season_payload('invalid')
        assert not rows

    def test_non_dict_items_skipped(self):
        """Non-dict items inside a list are silently skipped."""
        rows = TeamAnalyzer._normalize_team_season_payload(
            [None, 42, {'Name': 'Arsenal FC'}]
        )
        assert len(rows) == 1
# pylint: enable=protected-access,redefined-outer-name


# ---------------------------------------------------------------------------
# TeamAnalyzer._extract_team_season_row
# ---------------------------------------------------------------------------

# pylint: disable=protected-access
class TestExtractTeamSeasonRow:
    """Tests for the TeamAnalyzer._extract_team_season_row static method."""

    def test_returns_none_for_wrong_team(self):
        """A stats dict for a different team returns None."""
        stats = {'Name': 'Chelsea FC', 'Season': 2026}
        assert TeamAnalyzer._extract_team_season_row(stats, 'Arsenal FC') is None

    def test_returns_row_for_correct_team(self):
        """A stats dict for the requested team returns a populated row."""
        stats = {'Name': 'Arsenal FC', 'Season': 2026, 'Possession': 60.0}
        row = TeamAnalyzer._extract_team_season_row(stats, 'Arsenal FC')
        assert row is not None
        assert row['season_year'] == 2026

    def test_alternate_name_key_team_name(self):
        """TeamName is accepted as an alternate key for the team name."""
        stats = {'TeamName': 'Arsenal FC', 'Season': 2026}
        row = TeamAnalyzer._extract_team_season_row(stats, 'Arsenal FC')
        assert row is not None

    def test_case_insensitive_team_match(self):
        """Team name matching is case-insensitive."""
        stats = {'Name': 'arsenal fc', 'Season': 2026}
        row = TeamAnalyzer._extract_team_season_row(stats, 'Arsenal FC')
        assert row is not None
# pylint: enable=protected-access


# ---------------------------------------------------------------------------
# TeamAnalyzer._default_match_data
# ---------------------------------------------------------------------------

# pylint: disable=protected-access
class TestDefaultMatchData:
    """Tests for the TeamAnalyzer._default_match_data static method."""

    def test_returns_dataframe(self):
        """The fallback method returns a pandas DataFrame."""
        df = TeamAnalyzer._default_match_data()
        assert isinstance(df, pd.DataFrame)

    def test_has_two_rows(self):
        """The fallback data contains exactly two season rows."""
        df = TeamAnalyzer._default_match_data()
        assert len(df) == 2

    def test_custom_team_name(self):
        """A custom team name is applied to all rows."""
        df = TeamAnalyzer._default_match_data('Chelsea FC')
        assert (df['team'] == 'Chelsea FC').all()

    def test_seasons_present(self):
        """Both expected season years are present in the fallback data."""
        df = TeamAnalyzer._default_match_data()
        assert set(df['season_year']) == {2025, 2026}
# pylint: enable=protected-access


# ---------------------------------------------------------------------------
# TeamAnalyzer.calculate_basic_stats
# ---------------------------------------------------------------------------

# pylint: disable=redefined-outer-name
class TestCalculateBasicStats:
    """Tests for TeamAnalyzer.calculate_basic_stats."""

    def test_avg_possession_correct(self, loaded_analyzer):
        """Average possession is the mean of all rows for the team."""
        stats = loaded_analyzer.calculate_basic_stats('Arsenal FC')
        assert stats['avg_possession'] == pytest.approx(57.5)

    def test_shot_accuracy_correct(self, loaded_analyzer):
        """Shot accuracy is the mean of per-row on-target ratios."""
        # 2025: 80/200 = 40 %; 2026: 60/180 = 33.33 % → mean ≈ 36.67 %
        stats = loaded_analyzer.calculate_basic_stats('Arsenal FC')
        assert stats['shot_accuracy'] == pytest.approx(36.67, abs=0.01)

    def test_filtered_by_season(self, loaded_analyzer):
        """Passing season_year restricts the calculation to that season."""
        stats = loaded_analyzer.calculate_basic_stats('Arsenal FC', season_year=2025)
        assert stats['avg_possession'] == pytest.approx(60.0)

    def test_returns_nan_for_unknown_team(self, loaded_analyzer):
        """An unknown team name produces NaN for both metrics."""
        stats = loaded_analyzer.calculate_basic_stats('Unknown FC')
        assert _is_nan(stats['avg_possession'])
        assert _is_nan(stats['shot_accuracy'])

    def test_returns_nan_when_no_shots(self, analyzer):
        """Zero shots in every row produces NaN for shot accuracy."""
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
        """Missing possession column produces NaN for avg_possession."""
        analyzer.match_data = pd.DataFrame({
            'season_year': [2026],
            'team': ['Arsenal FC'],
            'shots': [100],
            'shots_on_target': [40],
        })
        stats = analyzer.calculate_basic_stats('Arsenal FC')
        assert _is_nan(stats['avg_possession'])

    def test_isolates_team_from_multi_team_data(self, two_team_analyzer):
        """Each team's stats are calculated independently."""
        arsenal = two_team_analyzer.calculate_basic_stats('Arsenal FC')
        chelsea = two_team_analyzer.calculate_basic_stats('Chelsea FC')
        assert arsenal['avg_possession'] != chelsea['avg_possession']
        assert arsenal['avg_possession'] == pytest.approx(60.0)
        assert chelsea['avg_possession'] == pytest.approx(50.0)
# pylint: enable=redefined-outer-name


# ---------------------------------------------------------------------------
# _build_possession_training_frame
# ---------------------------------------------------------------------------

class TestBuildPossessionTrainingFrame:
    """Tests for the pairwise possession training-data helper."""

    def test_builds_pairwise_rows(self):
        """Two teams produce two directed pairwise rows."""
        frame = _build_possession_training_frame({'A': 60.0, 'B': 40.0})
        assert len(frame) == 2
        assert set(frame.columns) == {
            'team_possession', 'opponent_possession',
            'possession_gap', 'target_possession',
        }

    def test_target_is_percentage_share(self):
        """Target possession is computed as team/(team+opponent)*100."""
        frame = _build_possession_training_frame({'A': 60.0, 'B': 40.0})
        row = frame.iloc[0]
        expected = row['team_possession'] / (
            row['team_possession'] + row['opponent_possession']
        ) * 100.0
        assert row['target_possession'] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# TeamAnalyzer.predict_next_rounds_possession
# ---------------------------------------------------------------------------

# pylint: disable=redefined-outer-name
class TestPredictNextRoundsPossession:
    """Tests for TeamAnalyzer.predict_next_rounds_possession."""

    def test_returns_predictions_for_each_opponent(self, two_team_analyzer):
        """One row is returned per requested opponent."""
        result = two_team_analyzer.predict_next_rounds_possession(
            'Arsenal FC', ['Chelsea FC']
        )
        assert len(result) == 1
        assert result.iloc[0]['opponent'] == 'Chelsea FC'

    def test_predicted_split_sums_to_100(self, two_team_analyzer):
        """Predicted team and opponent possession always sum to 100%."""
        result = two_team_analyzer.predict_next_rounds_possession(
            'Arsenal FC', ['Chelsea FC']
        )
        row = result.iloc[0]
        assert (
            row['predicted_team_possession'] + row['predicted_opponent_possession']
        ) == pytest.approx(100.0)

    def test_unknown_team_raises(self, two_team_analyzer):
        """A missing team in the dataset raises a ValueError."""
        with pytest.raises(ValueError, match='not found'):
            two_team_analyzer.predict_next_rounds_possession(
                'Unknown FC', ['Chelsea FC']
            )

    def test_fallback_when_sklearn_missing(self, two_team_analyzer, monkeypatch):
        """When sklearn is unavailable, fallback mode still returns predictions."""
        original_import = builtins.__import__

        def _failing_import(name, *args, **kwargs):
            if name == 'sklearn.linear_model':
                raise ImportError('sklearn missing')
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', _failing_import)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = two_team_analyzer.predict_next_rounds_possession(
                'Arsenal FC', ['Chelsea FC']
            )
        assert len(result) == 1
        assert any('Falling back to ratio-based' in str(w.message) for w in caught)
# pylint: enable=redefined-outer-name


# ---------------------------------------------------------------------------
# TeamAnalyzer.generate_report
# ---------------------------------------------------------------------------

# pylint: disable=redefined-outer-name
class TestGenerateReport:
    """Tests for TeamAnalyzer.generate_report."""

    def test_contains_team_name(self, loaded_analyzer):
        """The team name appears in the report output."""
        report = loaded_analyzer.generate_report('Arsenal FC', seasons=(2025, 2026))
        assert 'Arsenal FC' in report

    def test_contains_both_seasons(self, loaded_analyzer):
        """Both season years appear as column headers."""
        report = loaded_analyzer.generate_report('Arsenal FC', seasons=(2025, 2026))
        assert '2025' in report
        assert '2026' in report

    def test_contains_possession_label(self, loaded_analyzer):
        """The possession metric label is present."""
        report = loaded_analyzer.generate_report('Arsenal FC', seasons=(2025, 2026))
        assert 'Avg Possession' in report

    def test_contains_shot_accuracy_label(self, loaded_analyzer):
        """The shot accuracy metric label is present."""
        report = loaded_analyzer.generate_report('Arsenal FC', seasons=(2025, 2026))
        assert 'Shot Accuracy' in report

    def test_no_data_message_when_season_absent(self, loaded_analyzer):
        """A season not in match_data triggers the no-data message."""
        report = loaded_analyzer.generate_report('Arsenal FC', seasons=(2099,))
        assert 'No data available' in report

    def test_percentage_symbol_present(self, loaded_analyzer):
        """Percentage values are suffixed with the % symbol."""
        report = loaded_analyzer.generate_report('Arsenal FC', seasons=(2025, 2026))
        assert '%' in report
# pylint: enable=redefined-outer-name


# ---------------------------------------------------------------------------
# TeamAnalyzer.load_sample_data — fallback path
# ---------------------------------------------------------------------------

# pylint: disable=protected-access,redefined-outer-name
class TestLoadSampleDataFallback:
    """Tests for the no-API-key fallback path of load_sample_data."""

    def test_uses_fallback_when_no_api_key(self, analyzer):
        """Fallback data is loaded when no API key is configured."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('SPORTSDATA_API_KEY', None)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                analyzer.load_sample_data(api_key=None)
        assert analyzer.match_data is not None
        assert len(analyzer.match_data) == 2

    def test_fallback_has_correct_team(self, analyzer):
        """The custom team name is applied to the fallback data."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('SPORTSDATA_API_KEY', None)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                analyzer.load_sample_data(api_key=None, team_name='Chelsea FC')
        assert (analyzer.match_data['team'] == 'Chelsea FC').all()

    def test_warns_when_no_key(self, analyzer):
        """A RuntimeWarning mentioning SPORTSDATA_API_KEY is emitted."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(TeamAnalyzer, '_load_env_file'):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    analyzer.load_sample_data(api_key=None)
        messages = [str(w.message) for w in caught]
        assert any('SPORTSDATA_API_KEY' in m for m in messages)
# pylint: enable=protected-access,redefined-outer-name


# ---------------------------------------------------------------------------
# TeamAnalyzer.load_sample_data — API path (mocked)
# ---------------------------------------------------------------------------

# pylint: disable=redefined-outer-name
class TestLoadSampleDataApi:
    """Tests for the live-API path of load_sample_data (network mocked)."""

    def _make_payload(self, team_name, season):
        """Return a minimal single-team API payload for the given season."""
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
        """Data from two seasons is combined into match_data."""
        payloads = [
            self._make_payload('Arsenal FC', 2025),
            self._make_payload('Arsenal FC', 2026),
        ]
        with patch.object(TeamAnalyzer, '_fetch_json', side_effect=payloads):
            analyzer.load_sample_data(api_key='fake-key', team_name='Arsenal FC')
        assert set(analyzer.match_data['season_year']) == {2025, 2026}

    def test_skips_failed_season_with_warning(self, analyzer):
        """A season that raises is skipped and a warning is emitted."""
        def fail_first(url, **_):
            """Raise for 2025, succeed for 2026."""
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
        """If every season fails, fallback data is loaded instead."""
        with patch.object(TeamAnalyzer, '_fetch_json', side_effect=URLError('timeout')):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                analyzer.load_sample_data(api_key='fake-key', seasons=(2025, 2026))
        assert analyzer.match_data is not None

    def test_data_sorted_by_date(self, analyzer):
        """Combined match_data is sorted by date in ascending order."""
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
# pylint: enable=redefined-outer-name


# ---------------------------------------------------------------------------
# TeamAnalyzer.load_match_data_from_api
# ---------------------------------------------------------------------------

# pylint: disable=redefined-outer-name
class TestLoadMatchDataFromApi:
    """Tests for TeamAnalyzer.load_match_data_from_api."""

    def test_raises_without_api_key(self, analyzer):
        """Missing API key raises ValueError."""
        with pytest.raises(ValueError, match='API key'):
            analyzer.load_match_data_from_api(
                api_url='https://example.com', api_key=None
            )

    def test_raises_when_team_not_in_payload(self, analyzer, raw_season_stats_payload):
        """Team name absent from the payload raises ValueError."""
        with patch.object(TeamAnalyzer, '_fetch_json', return_value=raw_season_stats_payload):
            with pytest.raises(ValueError, match='Unknown FC'):
                analyzer.load_match_data_from_api(
                    api_url='https://example.com',
                    api_key='key',
                    team_name='Unknown FC',
                )

    def test_returns_dataframe_for_valid_team(self, analyzer, raw_season_stats_payload):
        """A valid response returns a one-row DataFrame for the requested team."""
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
        """Numeric columns are coerced away from object dtype after loading."""
        with patch.object(TeamAnalyzer, '_fetch_json', return_value=raw_season_stats_payload):
            df = analyzer.load_match_data_from_api(
                api_url='https://example.com', api_key='key', team_name='Arsenal FC'
            )
        assert pd.api.types.is_float_dtype(df['possession'])
        assert pd.api.types.is_numeric_dtype(df['shots'])

    def test_extracts_key_from_url(self, analyzer, raw_season_stats_payload):
        """A key embedded in the URL is accepted when api_key is None."""
        url = 'https://example.com?key=embedded-key'
        with patch.object(TeamAnalyzer, '_fetch_json', return_value=raw_season_stats_payload):
            df = analyzer.load_match_data_from_api(
                api_url=url, api_key=None, team_name='Arsenal FC'
            )
        assert df is not None
# pylint: enable=redefined-outer-name


# ---------------------------------------------------------------------------
# TeamAnalyzer.fetch_competition_details
# ---------------------------------------------------------------------------

# pylint: disable=redefined-outer-name
class TestFetchCompetitionDetails:
    """Tests for TeamAnalyzer.fetch_competition_details."""

    def test_returns_empty_without_key(self, analyzer):
        """No API key configured returns an empty dict."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('SPORTSDATA_API_KEY', None)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                result = analyzer.fetch_competition_details(api_key=None)
        assert result == {}

    def test_parses_teams_key(self, analyzer):
        """Teams list is parsed into a name → logo_url mapping."""
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
        """A network error returns an empty dict without raising."""
        with patch.object(TeamAnalyzer, '_fetch_json', side_effect=URLError('timeout')):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                result = analyzer.fetch_competition_details(api_key='key')
        assert result == {}

    def test_falls_back_to_team_name_key(self, analyzer):
        """TeamName is accepted as an alternate key for the team name."""
        payload = {'Teams': [{'TeamName': 'Juventus', 'WikipediaLogoUrl': None}]}
        with patch.object(TeamAnalyzer, '_fetch_json', return_value=payload):
            result = analyzer.fetch_competition_details(api_key='key')
        assert 'Juventus' in result
# pylint: enable=redefined-outer-name


# ---------------------------------------------------------------------------
# TeamAnalyzer.fetch_all_teams
# ---------------------------------------------------------------------------

# pylint: disable=redefined-outer-name
class TestFetchAllTeams:
    """Tests for TeamAnalyzer.fetch_all_teams."""

    def test_returns_fallback_without_key(self, analyzer):
        """No API key returns a list containing only the default team name."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('SPORTSDATA_API_KEY', None)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                teams = analyzer.fetch_all_teams(api_key=None)
        assert teams == [TeamAnalyzer.default_team_name]

    def test_returns_sorted_deduplicated_list(self, analyzer, raw_season_stats_payload):
        """Returned team list is sorted and contains no duplicates."""
        with patch.object(TeamAnalyzer, '_fetch_json', return_value=raw_season_stats_payload):
            teams = analyzer.fetch_all_teams(api_key='key', seasons=(2026,))
        assert teams == sorted(teams)
        assert len(teams) == len(set(teams))
        assert 'Arsenal FC' in teams
        assert 'Chelsea FC' in teams

    def test_deduplicates_across_seasons(self, analyzer, raw_season_stats_payload):
        """Teams appearing in multiple seasons are listed only once."""
        with patch.object(
            TeamAnalyzer, '_fetch_json', return_value=raw_season_stats_payload
        ):
            teams = analyzer.fetch_all_teams(api_key='key', seasons=(2025, 2026))
        assert teams.count('Arsenal FC') == 1

    def test_skips_failed_season(self, analyzer, raw_season_stats_payload):
        """A failing season is skipped; teams from other seasons are returned."""
        def fail_first(url, **_):
            """Raise for 2025, succeed for 2026."""
            if '2025' in url:
                raise URLError('timeout')
            return raw_season_stats_payload

        with patch.object(TeamAnalyzer, '_fetch_json', side_effect=fail_first):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                teams = analyzer.fetch_all_teams(api_key='key', seasons=(2025, 2026))
        assert 'Arsenal FC' in teams

    def test_returns_fallback_when_all_seasons_fail(self, analyzer):
        """If every season fails, the default team list is returned."""
        with patch.object(TeamAnalyzer, '_fetch_json', side_effect=URLError('timeout')):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                teams = analyzer.fetch_all_teams(api_key='key', seasons=(2026,))
        assert teams == [TeamAnalyzer.default_team_name]
# pylint: enable=redefined-outer-name


# ---------------------------------------------------------------------------
# TeamAnalyzer.plot_performance_trends
# ---------------------------------------------------------------------------

# pylint: disable=redefined-outer-name
class TestPlotPerformanceTrends:
    """Tests for TeamAnalyzer.plot_performance_trends."""

    def test_returns_figure(self, loaded_analyzer):
        """The method returns a matplotlib Figure object."""
        fig = loaded_analyzer.plot_performance_trends('Arsenal FC', seasons=(2025, 2026))
        assert fig is not None
        plt.close(fig)

    def test_figure_has_two_axes(self, loaded_analyzer):
        """The figure contains exactly two subplot axes."""
        fig = loaded_analyzer.plot_performance_trends('Arsenal FC', seasons=(2025, 2026))
        subplots = [ax for ax in fig.axes if ax.get_subplotspec() is not None]
        assert len(subplots) == 2
        plt.close(fig)

    def test_suptitle_contains_team_name(self, loaded_analyzer):
        """The figure suptitle includes the team name."""
        fig = loaded_analyzer.plot_performance_trends('Arsenal FC')
        assert 'Arsenal FC' in fig.texts[0].get_text()
        plt.close(fig)

    def test_logo_axes_added_when_logo_present(self, loaded_analyzer):
        """A logo URL causes a third logo axes to be added to the figure."""
        fake_logo = Image.new('RGBA', (52, 52), color=(255, 0, 0, 255))
        with patch('team_analyser._fetch_logo_pil', return_value=fake_logo):
            fig = loaded_analyzer.plot_performance_trends(
                'Arsenal FC', logo_url='https://example.com/logo.png'
            )
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_no_logo_axes_without_logo(self, loaded_analyzer):
        """A None logo result leaves the figure with exactly two axes."""
        with patch('team_analyser._fetch_logo_pil', return_value=None):
            fig = loaded_analyzer.plot_performance_trends(
                'Arsenal FC', logo_url='https://example.com/logo.png'
            )
        assert len(fig.axes) == 2
        plt.close(fig)
# pylint: enable=redefined-outer-name


# ---------------------------------------------------------------------------
# TeamAnalyzer._log_raw_api_data
# ---------------------------------------------------------------------------

# pylint: disable=protected-access
class TestLogRawApiData:
    """Tests for the TeamAnalyzer._log_raw_api_data static method."""

    def test_creates_json_file(self, tmp_path):
        """Calling the method creates the specified JSON file."""
        TeamAnalyzer._log_buffer = {}
        log_file = tmp_path / 'test_log.json'
        TeamAnalyzer._log_raw_api_data(
            2026, [{'Name': 'Arsenal FC'}], 'Arsenal FC', log_path=log_file
        )
        assert log_file.exists()

    def test_json_contains_season_data(self, tmp_path):
        """The written JSON contains the season entry with correct row_count."""
        TeamAnalyzer._log_buffer = {}
        log_file = tmp_path / 'test_log.json'
        TeamAnalyzer._log_raw_api_data(
            2026, [{'Name': 'Arsenal FC'}], 'Arsenal FC', log_path=log_file
        )
        data = json.loads(log_file.read_text())
        assert '2026' in data['api_log']
        assert data['api_log']['2026']['row_count'] == 1

    def test_accumulates_multiple_seasons(self, tmp_path):
        """Successive calls with the same path accumulate seasons in one file."""
        TeamAnalyzer._log_buffer = {}
        log_file = tmp_path / 'test_log.json'
        TeamAnalyzer._log_raw_api_data(2025, [], 'Arsenal FC', log_path=log_file)
        TeamAnalyzer._log_raw_api_data(2026, [], 'Arsenal FC', log_path=log_file)
        data = json.loads(log_file.read_text())
        assert '2025' in data['api_log']
        assert '2026' in data['api_log']
# pylint: enable=protected-access


# ---------------------------------------------------------------------------
# TeamAnalyzer._load_env_file
# ---------------------------------------------------------------------------

# pylint: disable=protected-access
class TestLoadEnvFile:
    """Tests for the TeamAnalyzer._load_env_file static method."""

    def test_loads_key_from_file(self, tmp_path, monkeypatch):
        """A KEY=value line is loaded into the environment."""
        env_file = tmp_path / '.env'
        env_file.write_text('SPORTSDATA_API_KEY=test-key-123\n')
        monkeypatch.delenv('SPORTSDATA_API_KEY', raising=False)
        TeamAnalyzer._load_env_file(str(env_file))
        assert os.environ.get('SPORTSDATA_API_KEY') == 'test-key-123'

    def test_strips_quotes(self, tmp_path, monkeypatch):
        """Surrounding quotes are stripped from values."""
        env_file = tmp_path / '.env'
        env_file.write_text('SPORTSDATA_API_KEY="quoted-key"\n')
        monkeypatch.delenv('SPORTSDATA_API_KEY', raising=False)
        TeamAnalyzer._load_env_file(str(env_file))
        assert os.environ['SPORTSDATA_API_KEY'] == 'quoted-key'

    def test_ignores_comments(self, tmp_path, monkeypatch):
        """Lines starting with # are ignored."""
        env_file = tmp_path / '.env'
        env_file.write_text('# comment\nSPORTSDATA_API_KEY=real-key\n')
        monkeypatch.delenv('SPORTSDATA_API_KEY', raising=False)
        TeamAnalyzer._load_env_file(str(env_file))
        assert os.environ['SPORTSDATA_API_KEY'] == 'real-key'

    def test_does_not_override_existing_env(self, tmp_path, monkeypatch):
        """An already-set environment variable is not overwritten."""
        env_file = tmp_path / '.env'
        env_file.write_text('SPORTSDATA_API_KEY=from-file\n')
        monkeypatch.setenv('SPORTSDATA_API_KEY', 'already-set')
        TeamAnalyzer._load_env_file(str(env_file))
        assert os.environ['SPORTSDATA_API_KEY'] == 'already-set'

    def test_missing_file_does_not_raise(self):
        """A path that does not exist returns silently."""
        TeamAnalyzer._load_env_file('/nonexistent/.env')
# pylint: enable=protected-access
