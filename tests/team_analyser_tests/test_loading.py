"""Tests for API loading and team discovery behaviour."""

import warnings
from unittest.mock import patch
from urllib.error import URLError

import pandas as pd
import pytest

from team_analyser import TeamAnalyzer


class TestLoadSampleDataFallback:
    """Tests for the no-API-key fallback path of load_sample_data."""

    def test_uses_fallback_when_no_api_key(self, analyzer, without_api_key):
        with without_api_key():
            analyzer.load_sample_data(api_key=None)
        assert analyzer.match_data is not None
        assert len(analyzer.match_data) == 2

    def test_fallback_has_correct_team(self, analyzer, without_api_key):
        with without_api_key():
            analyzer.load_sample_data(api_key=None, team_name='Chelsea FC')
        assert (analyzer.match_data['team'] == 'Chelsea FC').all()

    def test_warns_when_no_key(self, analyzer, without_api_key):
        with patch.object(TeamAnalyzer, '_load_env_file'):
            with without_api_key() as caught:
                analyzer.load_sample_data(api_key=None)
        messages = [str(warning.message) for warning in caught]
        assert any('SPORTSDATA_API_KEY' in message for message in messages)


class TestLoadSampleDataApi:
    """Tests for the live-API path of load_sample_data with network mocked."""

    def test_loads_two_seasons(self, analyzer, make_season_payload):
        payloads = [
            make_season_payload('Arsenal FC', 2025),
            make_season_payload('Arsenal FC', 2026),
        ]
        with patch.object(TeamAnalyzer, '_fetch_json', side_effect=payloads):
            analyzer.load_sample_data(api_key='fake-key', team_name='Arsenal FC')
        assert set(analyzer.match_data['season_year']) == {2025, 2026}

    def test_skips_failed_season_with_warning(self, analyzer, make_season_payload):
        def fail_first(url, **_):
            if '2025' in url:
                raise ValueError('API error')
            return make_season_payload('Arsenal FC', 2026)

        with patch.object(TeamAnalyzer, '_fetch_json', side_effect=fail_first):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                analyzer.load_sample_data(
                    api_key='fake-key', team_name='Arsenal FC', seasons=(2025, 2026)
                )
        assert len(analyzer.match_data) == 1
        assert any('2025' in str(warning.message) for warning in caught)

    def test_falls_back_when_all_seasons_fail(self, analyzer):
        with patch.object(TeamAnalyzer, '_fetch_json', side_effect=URLError('timeout')):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                analyzer.load_sample_data(api_key='fake-key', seasons=(2025, 2026))
        assert analyzer.match_data is not None

    def test_data_sorted_by_date(self, analyzer, make_season_payload):
        payloads = [
            make_season_payload('Arsenal FC', 2026),
            make_season_payload('Arsenal FC', 2025),
        ]
        with patch.object(TeamAnalyzer, '_fetch_json', side_effect=payloads):
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
        assert pd.api.types.is_numeric_dtype(df['shots'])

    def test_extracts_key_from_url(self, analyzer, raw_season_stats_payload):
        url = 'https://example.com?key=embedded-key'
        with patch.object(TeamAnalyzer, '_fetch_json', return_value=raw_season_stats_payload):
            df = analyzer.load_match_data_from_api(
                api_url=url, api_key=None, team_name='Arsenal FC'
            )
        assert df is not None


class TestFetchCompetitionDetails:
    """Tests for TeamAnalyzer.fetch_competition_details."""

    def test_returns_empty_without_key(self, analyzer, without_api_key):
        with without_api_key():
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


class TestFetchAllTeams:
    """Tests for TeamAnalyzer.fetch_all_teams."""

    def test_returns_fallback_without_key(self, analyzer, without_api_key):
        with without_api_key():
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
        with patch.object(TeamAnalyzer, '_fetch_json', return_value=raw_season_stats_payload):
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
        assert 'Arsenal FC' in teams

    def test_returns_fallback_when_all_seasons_fail(self, analyzer):
        with patch.object(TeamAnalyzer, '_fetch_json', side_effect=URLError('timeout')):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                teams = analyzer.fetch_all_teams(api_key='key', seasons=(2026,))
        assert teams == [TeamAnalyzer.default_team_name]
