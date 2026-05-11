"""Tests for small helper methods in team_analyser.py."""
# pylint: disable=missing-function-docstring,protected-access

import numpy as np
import pandas as pd

from team_analyser import TeamAnalyzer, _build_api_headers, _coerce_numeric_cols, _is_nan


class TestIsNan:
    """Tests for the _is_nan module-level helper."""

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


class TestBuildApiHeaders:
    """Tests for the _build_api_headers module-level helper."""

    def test_returns_dict_with_key(self):
        headers = _build_api_headers('my-key')
        assert headers['Ocp-Apim-Subscription-Key'] == 'my-key'

    def test_accept_header_present(self):
        headers = _build_api_headers('x')
        assert headers['Accept'] == 'application/json'


class TestCoerceNumericCols:
    """Tests for the _coerce_numeric_cols module-level helper."""

    def test_converts_string_numbers(self):
        df = pd.DataFrame({'possession': ['60.5', '55.0'], 'shots': ['200', '150']})
        _coerce_numeric_cols(df)
        assert pd.api.types.is_float_dtype(df['possession'])
        assert pd.api.types.is_numeric_dtype(df['shots'])

    def test_ignores_missing_columns(self):
        df = pd.DataFrame({'possession': [60.5]})
        _coerce_numeric_cols(df)
        assert df['possession'].dtype == float

    def test_coerces_invalid_to_nan(self):
        df = pd.DataFrame({'possession': ['not-a-number']})
        _coerce_numeric_cols(df)
        assert np.isnan(df['possession'].iloc[0])


class TestFmt:
    """Tests for the TeamAnalyzer._fmt static formatting helper."""

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


class TestPickValue:
    """Tests for the TeamAnalyzer._pick_value static helper."""

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


class TestTeamMatches:
    """Tests for the TeamAnalyzer._team_matches static helper."""

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


class TestExtractApiKeyFromUrl:
    """Tests for the TeamAnalyzer._extract_api_key_from_url static helper."""

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


class TestDefaultMatchData:
    """Tests for the TeamAnalyzer._default_match_data static method."""

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
