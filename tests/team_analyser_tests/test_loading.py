"""Tests for API loading and team discovery behaviour."""
# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import warnings
from urllib.error import URLError

import pandas as pd
import pytest

from team_analyser import TeamAnalyzer

from .assertions import (
    assert_default_rows_loaded,
    assert_row_count,
    assert_season_years,
    assert_single_season_frame,
    assert_sorted_unique_teams,
    assert_warning_mentions,
)


def _load_api_payload(analyzer, patched_fetch_json, payload, team_name='Arsenal FC'):
    """Load one API payload into a DataFrame for a team."""
    with patched_fetch_json(return_value=payload):
        return analyzer.load_match_data_from_api(
            api_url='https://example.com', api_key='key', team_name=team_name
        )


def _fetch_teams(analyzer, patched_fetch_json, payload, seasons=(2026,)):
    """Fetch team names from a patched TeamSeasonStats payload."""
    with patched_fetch_json(return_value=payload):
        return analyzer.fetch_all_teams(api_key='key', seasons=seasons)


class TestLoadSampleDataFallback:
    """Tests for the no-API-key fallback path of load_sample_data."""

    def test_uses_fallback_when_no_api_key(self, analyzer, without_api_key):
        with without_api_key():
            analyzer.load_sample_data(api_key=None)
        assert_default_rows_loaded(analyzer)

    def test_fallback_has_correct_team(self, analyzer, without_api_key):
        with without_api_key():
            analyzer.load_sample_data(api_key=None, team_name='Chelsea FC')
        assert_default_rows_loaded(analyzer, expected_team='Chelsea FC')

    def test_warns_when_no_key(self, analyzer, patched_fetch_json, without_api_key):
        with patched_fetch_json():
            with without_api_key() as caught:
                analyzer.load_sample_data(api_key=None)
        assert_warning_mentions(caught, 'SPORTSDATA_API_KEY')


class TestLoadSampleDataApi:
    """Tests for the live-API path of load_sample_data with network mocked."""

    def test_loads_two_seasons(self, analyzer, make_season_payload, patched_fetch_json):
        payloads = [
            make_season_payload('Arsenal FC', 2025),
            make_season_payload('Arsenal FC', 2026),
        ]
        with patched_fetch_json(side_effect=payloads):
            analyzer.load_sample_data(api_key='fake-key', team_name='Arsenal FC')
        assert_season_years(analyzer.match_data, {2025, 2026})

    def test_skips_failed_season_with_warning(
        self, analyzer, make_season_payload, patched_fetch_json
    ):
        def fail_first(url, **_):
            if '2025' in url:
                raise ValueError('API error')
            return make_season_payload('Arsenal FC', 2026)

        with patched_fetch_json(side_effect=fail_first):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                analyzer.load_sample_data(
                    api_key='fake-key', team_name='Arsenal FC', seasons=(2025, 2026)
                )
        assert_row_count(analyzer.match_data, 1)
        assert_warning_mentions(caught, '2025')

    def test_falls_back_when_all_seasons_fail(self, analyzer, patched_fetch_json):
        with patched_fetch_json(side_effect=URLError('timeout')):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                analyzer.load_sample_data(api_key='fake-key', seasons=(2025, 2026))
        assert_default_rows_loaded(analyzer)

    def test_data_sorted_by_date(self, analyzer, make_season_payload, patched_fetch_json):
        payloads = [
            make_season_payload('Arsenal FC', 2026),
            make_season_payload('Arsenal FC', 2025),
        ]
        with patched_fetch_json(side_effect=payloads):
            analyzer.load_sample_data(
                api_key='fake-key', team_name='Arsenal FC', seasons=(2026, 2025)
            )
        years = list(analyzer.match_data['season_year'])
        assert years == sorted(years)


class TestLoadMatchDataFromApi:
    """Tests for TeamAnalyzer.load_match_data_from_api."""

    def test_raises_without_api_key(self, analyzer):
        with pytest.raises(ValueError, match='API key'):
            analyzer.load_match_data_from_api(
                api_url='https://example.com', api_key=None
            )

    def test_raises_when_team_not_in_payload(
        self, analyzer, raw_season_stats_payload, patched_fetch_json
    ):
        with patched_fetch_json(return_value=raw_season_stats_payload):
            with pytest.raises(ValueError, match='Unknown FC'):
                analyzer.load_match_data_from_api(
                    api_url='https://example.com',
                    api_key='key',
                    team_name='Unknown FC',
                )

    def test_returns_dataframe_for_valid_team(
        self, analyzer, raw_season_stats_payload, patched_fetch_json
    ):
        df = _load_api_payload(analyzer, patched_fetch_json, raw_season_stats_payload)
        assert isinstance(df, pd.DataFrame)
        assert_single_season_frame(df, 2026)

    def test_numeric_cols_are_float(self, analyzer, raw_season_stats_payload, patched_fetch_json):
        df = _load_api_payload(analyzer, patched_fetch_json, raw_season_stats_payload)
        assert pd.api.types.is_float_dtype(df['possession'])
        assert pd.api.types.is_numeric_dtype(df['shots'])

    def test_extracts_key_from_url(self, analyzer, raw_season_stats_payload, patched_fetch_json):
        with patched_fetch_json(return_value=raw_season_stats_payload):
            df = analyzer.load_match_data_from_api(
                api_url='https://example.com?key=embedded-key',
                api_key=None,
                team_name='Arsenal FC',
            )
        assert df is not None


class TestFetchCompetitionDetails:
    """Tests for TeamAnalyzer.fetch_competition_details."""

    def test_returns_empty_without_key(self, analyzer, without_api_key):
        with without_api_key():
            result = analyzer.fetch_competition_details(api_key=None)
        assert result == {}

    def test_parses_teams_key(self, analyzer, patched_fetch_json):
        payload = {
            'Teams': [
                {'Name': 'Arsenal FC', 'WikipediaLogoUrl': 'https://upload.wikimedia.org/...'},
                {'Name': 'Chelsea FC', 'WikipediaLogoUrl': None},
            ]
        }
        with patched_fetch_json(return_value=payload):
            result = analyzer.fetch_competition_details(api_key='key')
        assert result == {
            'Arsenal FC': 'https://upload.wikimedia.org/...',
            'Chelsea FC': None,
        }

    def test_returns_empty_on_network_error(self, analyzer, patched_fetch_json):
        with patched_fetch_json(side_effect=URLError('timeout')):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                result = analyzer.fetch_competition_details(api_key='key')
        assert result == {}

    def test_falls_back_to_team_name_key(self, analyzer, patched_fetch_json):
        payload = {'Teams': [{'TeamName': 'Juventus', 'WikipediaLogoUrl': None}]}
        with patched_fetch_json(return_value=payload):
            result = analyzer.fetch_competition_details(api_key='key')
        assert 'Juventus' in result


class TestFetchAllTeams:
    """Tests for TeamAnalyzer.fetch_all_teams."""

    def test_returns_fallback_without_key(self, analyzer, without_api_key):
        with without_api_key():
            teams = analyzer.fetch_all_teams(api_key=None)
        assert teams == [TeamAnalyzer.default_team_name]

    def test_returns_sorted_deduplicated_list(
        self, analyzer, raw_season_stats_payload, patched_fetch_json
    ):
        teams = _fetch_teams(analyzer, patched_fetch_json, raw_season_stats_payload)
        assert_sorted_unique_teams(teams, 'Arsenal FC', 'Chelsea FC')

    def test_deduplicates_across_seasons(
        self, analyzer, raw_season_stats_payload, patched_fetch_json
    ):
        teams = _fetch_teams(
            analyzer, patched_fetch_json, raw_season_stats_payload, seasons=(2025, 2026)
        )
        assert_row_count([team for team in teams if team == 'Arsenal FC'], 1)

    def test_skips_failed_season(self, analyzer, raw_season_stats_payload, patched_fetch_json):
        def fail_first(url, **_):
            if '2025' in url:
                raise URLError('timeout')
            return raw_season_stats_payload

        with patched_fetch_json(side_effect=fail_first):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                teams = analyzer.fetch_all_teams(api_key='key', seasons=(2025, 2026))
        assert 'Arsenal FC' in teams

    def test_returns_fallback_when_all_seasons_fail(self, analyzer, patched_fetch_json):
        with patched_fetch_json(side_effect=URLError('timeout')):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                teams = analyzer.fetch_all_teams(api_key='key', seasons=(2026,))
        assert teams == [TeamAnalyzer.default_team_name]
