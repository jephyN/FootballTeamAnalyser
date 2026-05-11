"""Tests for API log persistence and .env loading."""
# pylint: disable=missing-function-docstring,protected-access

import json
import os

from team_analyser import TeamAnalyzer


class TestLogRawApiData:
    """Tests for the TeamAnalyzer._log_raw_api_data static method."""

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


class TestLoadEnvFile:
    """Tests for the TeamAnalyzer._load_env_file static method."""

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
        TeamAnalyzer._load_env_file('/nonexistent/.env')
