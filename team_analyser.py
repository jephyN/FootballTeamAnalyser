import json
import os
import warnings
from datetime import datetime
from urllib.error import URLError
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TeamAnalyzer:
    DEFAULT_API_URL = 'https://api.sportsdata.io/v4/soccer/scores/json/TeamSeasonStats/3/2025?key=1527a55559834d689d6e2ad76e950fb4'
    DEFAULT_TEAM_NAME = 'Arsenal FC'

    def __init__(self):
        """Initialize the analyzer with empty data structures."""
        self.team_data = None
        self.match_data = None

    @staticmethod
    def _default_match_data():
        """Static fallback data used when API configuration is unavailable."""
        return pd.DataFrame({
            'date': pd.to_datetime(['2024-01-15', '2024-01-22', '2024-01-29', '2024-02-05', '2024-02-12']),
            'team': ['Arsenal FC'] * 5,
            'opponent': ['Liverpool', 'Chelsea', 'Manchester City', 'Tottenham', 'Newcastle'],
            'venue': ['Home', 'Away', 'Home', 'Away', 'Home'],
            'goals_scored': [2, 1, 3, 2, 4],
            'goals_conceded': [1, 2, 0, 1, 1],
            'possession': [55, 48, 62, 51, 58],
            'shots': [14, 11, 17, 13, 18],
            'shots_on_target': [6, 4, 8, 5, 9],
        })

    @staticmethod
    def _fetch_json(url, headers=None, params=None):
        """Fetch JSON payload from an HTTP endpoint."""
        if params:
            separator = '&' if '?' in url else '?'
            url = f"{url}{separator}{urlencode(params)}"

        req = Request(url, headers=headers or {})
        with urlopen(req, timeout=30) as response:
            payload = response.read().decode('utf-8', errors='ignore')
        return json.loads(payload)

    @staticmethod
    def _normalize_games_payload(payload):
        """Normalize SportsData.io game responses to a list of game-like dicts."""
        if isinstance(payload, list):
            return payload

        if isinstance(payload, dict):
            for key in ('games', 'Games', 'data', 'Data', 'matches', 'Matches'):
                value = payload.get(key)
                if isinstance(value, list):
                    return value

        return []

    @staticmethod
    def _normalize_team_season_payload(payload):
        """Normalize SportsData.io TeamSeasonStats responses to list form."""
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]

        if isinstance(payload, dict):
            for key in ('TeamSeasonStats', 'teamSeasonStats', 'data', 'Data'):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
                if isinstance(value, dict):
                    return [value]
            return [payload]

        return []

    @staticmethod
    def _load_env_file(env_path='.env'):
        """Load simple KEY=VALUE pairs from a local .env file into environment."""
        if not os.path.exists(env_path):
            return

        with open(env_path, 'r', encoding='utf-8') as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue

                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value

    @staticmethod
    def _extract_api_key_from_url(url):
        """Extract `key` query parameter from URL and return sanitized URL + key."""
        if not url:
            return url, None

        parsed = urlparse(url)
        query = parse_qs(parsed.query, keep_blank_values=True)
        values = query.pop('key', None)
        extracted_key = values[0] if values else None
        cleaned_query = urlencode(query, doseq=True)
        cleaned_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, cleaned_query, parsed.fragment))
        return cleaned_url, extracted_key

    @staticmethod
    def _pick_value(item, *keys):
        """Return the first non-empty value from a mapping for any candidate keys."""
        for key in keys:
            value = item.get(key)
            if value is not None and value != '':
                return value
        return np.nan

    @staticmethod
    def _team_matches(row_team_name, requested_team_name):
        """Case-insensitive matcher that supports Arsenal aliases."""
        if pd.isna(row_team_name):
            return False

        normalized = str(row_team_name).strip().lower()
        requested = str(requested_team_name).strip().lower()
        aliases = {'arsenal', 'arsenal fc', 'ars'}

        if requested in aliases:
            return normalized in aliases
        return normalized == requested

    @staticmethod
    def _extract_row(game, team_name):
        """Convert one SportsData.io game payload into the local match row schema."""
        home_team = TeamAnalyzer._pick_value(game, 'HomeTeamName', 'HomeTeam', 'HomeTeamKey', 'HomeTeamCode')
        away_team = TeamAnalyzer._pick_value(game, 'AwayTeamName', 'AwayTeam', 'AwayTeamKey', 'AwayTeamCode')

        if pd.isna(home_team) or pd.isna(away_team):
            return None

        home_name = str(home_team).strip()
        away_name = str(away_team).strip()
        is_team_home = TeamAnalyzer._team_matches(home_name, team_name)
        is_team_away = TeamAnalyzer._team_matches(away_name, team_name)
        if not (is_team_home or is_team_away):
            return None

        home_score = TeamAnalyzer._pick_value(game, 'HomeTeamScore', 'HomeScore', 'ScoreHome')
        away_score = TeamAnalyzer._pick_value(game, 'AwayTeamScore', 'AwayScore', 'ScoreAway')

        return {
            'date': pd.to_datetime(TeamAnalyzer._pick_value(game, 'Day', 'DateTime', 'MatchTime', 'Date'), errors='coerce'),
            'team': 'Arsenal FC',
            'opponent': away_name if is_team_home else home_name,
            'venue': 'Home' if is_team_home else 'Away',
            'goals_scored': home_score if is_team_home else away_score,
            'goals_conceded': away_score if is_team_home else home_score,
            'possession': TeamAnalyzer._pick_value(game, 'Possession', 'PossessionPct'),
            'shots': TeamAnalyzer._pick_value(game, 'Shots', 'TeamShots'),
            'shots_on_target': TeamAnalyzer._pick_value(game, 'ShotsOnGoal', 'ShotsOnTarget'),
        }

    @staticmethod
    def _extract_team_season_row(team_stats, team_name='Arsenal FC'):
        """Convert TeamSeasonStats payload into the local row schema."""
        source_team = TeamAnalyzer._pick_value(team_stats, 'Name', 'TeamName', 'Team', 'TeamKey')
        if not TeamAnalyzer._team_matches(source_team, team_name):
            return None

        season = TeamAnalyzer._pick_value(team_stats, 'Season', 'SeasonYear')
        season_year = pd.to_numeric(season, errors='coerce')
        season_date = pd.to_datetime(f"{int(season_year)}-12-31", errors='coerce') if not pd.isna(season_year) else pd.NaT

        return {
            'date': season_date,
            'team': 'Arsenal FC',
            'opponent': 'Season Aggregate',
            'venue': 'N/A',
            'goals_scored': TeamAnalyzer._pick_value(team_stats, 'Goals', 'GoalsScored', 'TeamGoals'),
            'goals_conceded': TeamAnalyzer._pick_value(team_stats, 'OpponentGoals', 'GoalsAgainst', 'GoalsConceded'),
            'possession': TeamAnalyzer._pick_value(team_stats, 'Possession', 'PossessionPct', 'PossessionPercentage'),
            'shots': TeamAnalyzer._pick_value(team_stats, 'Shots', 'ShotsTotal', 'TeamShots'),
            'shots_on_target': TeamAnalyzer._pick_value(team_stats, 'ShotsOnGoal', 'ShotsOnTarget'),
            'matches_in_sample': TeamAnalyzer._pick_value(team_stats, 'Games', 'GamesPlayed', 'Matches', 'MatchesPlayed'),
            'clean_sheets': TeamAnalyzer._pick_value(team_stats, 'CleanSheets'),
        }

    @classmethod
    def _default_api_url(cls):
        return cls.DEFAULT_API_URL

    def load_sample_data(self, api_key=None, api_url=None, team_name=DEFAULT_TEAM_NAME):
        """Load SportsData.io data when configured; otherwise use fallback data."""
        self._load_env_file()
        resolved_api_url = api_url or os.getenv('SPORTSDATA_MATCHES_URL') or self._default_api_url()
        resolved_api_url, key_from_url = self._extract_api_key_from_url(resolved_api_url)
        resolved_api_key = api_key or os.getenv('SPORTSDATA_API_KEY') or os.getenv('key') or key_from_url

        if not resolved_api_key:
            warnings.warn('SPORTSDATA_API_KEY is not configured. Using fallback sample data.', RuntimeWarning)
            self.match_data = self._default_match_data()
            return

        try:
            self.load_match_data_from_api(api_url=resolved_api_url, api_key=resolved_api_key, team_name=team_name)
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            warnings.warn(f'Failed to load SportsData.io data ({exc}). Using fallback sample data.', RuntimeWarning)
            self.match_data = self._default_match_data()

    def load_match_data_from_api(self, api_url, api_key, team_name=DEFAULT_TEAM_NAME):
        """Load matches from a SportsData.io endpoint and replace `match_data`."""
        cleaned_url, key_from_url = self._extract_api_key_from_url(api_url)
        resolved_api_key = api_key or key_from_url
        if not resolved_api_key:
            raise ValueError('A SportsData.io API key is required.')

        headers = {
            'Accept': 'application/json',
            'Ocp-Apim-Subscription-Key': resolved_api_key,
        }
        payload = self._fetch_json(cleaned_url, headers=headers)
        rows = []

        games = self._normalize_games_payload(payload)
        for game in games:
            if not isinstance(game, dict):
                continue
            row = self._extract_row(game, team_name=team_name)
            if row:
                rows.append(row)

        if not rows:
            season_rows = self._normalize_team_season_payload(payload)
            for season_row in season_rows:
                row = self._extract_team_season_row(season_row, team_name=team_name)
                if row:
                    rows.append(row)

        if not rows:
            raise ValueError('No Arsenal FC data parsed from SportsData.io response.')

        match_data = pd.DataFrame(rows)
        numeric_cols = ['goals_scored', 'goals_conceded', 'possession', 'shots', 'shots_on_target', 'matches_in_sample', 'clean_sheets']
        for col in numeric_cols:
            if col in match_data.columns:
                match_data[col] = pd.to_numeric(match_data[col], errors='coerce')

        self.match_data = match_data.sort_values('date').reset_index(drop=True)

    def calculate_basic_stats(self, team_name):
        """Calculate basic team statistics."""
        team_matches = self.match_data[self.match_data['team'] == team_name]

        if 'matches_in_sample' in team_matches.columns and team_matches['matches_in_sample'].notna().any():
            matches_played = int(pd.to_numeric(team_matches['matches_in_sample'], errors='coerce').sum())
        else:
            matches_played = len(team_matches)

        goals_scored = team_matches.get('goals_scored', pd.Series(dtype=float)).sum(min_count=1)
        goals_conceded = team_matches.get('goals_conceded', pd.Series(dtype=float)).sum(min_count=1)
        goals_per_game = (goals_scored / matches_played) if matches_played else np.nan

        if 'clean_sheets' in team_matches.columns and team_matches['clean_sheets'].notna().any():
            clean_sheets = int(pd.to_numeric(team_matches['clean_sheets'], errors='coerce').sum())
        elif 'goals_conceded' in team_matches:
            clean_sheets = len(team_matches[team_matches.get('goals_conceded', pd.Series(dtype=float)) == 0])
        else:
            clean_sheets = np.nan

        stats = {
            'matches_played': matches_played,
            'goals_scored': goals_scored,
            'goals_conceded': goals_conceded,
            'goal_difference': goals_scored - goals_conceded,
            'avg_possession': team_matches['possession'].mean() if 'possession' in team_matches else np.nan,
            'shot_accuracy': (team_matches['shots_on_target'].sum() / team_matches['shots'].sum() * 100) if {'shots_on_target', 'shots'}.issubset(team_matches.columns) else np.nan,
            'goals_per_game': goals_per_game,
            'clean_sheets': clean_sheets,
        }

        return pd.Series(stats)

    def plot_performance_trends(self, team_name):
        """Plot performance trends over time."""
        team_matches = self.match_data[self.match_data['team'] == team_name]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(
            team_matches['date'],
            team_matches.get('goals_scored', pd.Series(index=team_matches.index, dtype=float)),
            label='Goals Scored',
            marker='o',
        )
        ax1.plot(
            team_matches['date'],
            team_matches.get('goals_conceded', pd.Series(index=team_matches.index, dtype=float)),
            label='Goals Conceded',
            marker='o',
        )
        ax1.set_title(f'{team_name} Goal Performance Over Time')
        ax1.legend()
        ax1.grid(True)

        if 'possession' in team_matches:
            ax2.plot(team_matches['date'], team_matches['possession'], label='Possession %', marker='o', color='green')
            ax2.set_title(f'{team_name} Possession Over Time')
            ax2.legend()
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, 'Possession data is not available for this dataset', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_axis_off()

        plt.tight_layout()
        return fig

    def generate_report(self, team_name):
        """Generate a comprehensive performance report."""
        stats = self.calculate_basic_stats(team_name)

        report = f"""
Performance Report for {team_name}

Basic Statistics:
----------------
Matches Played: {stats['matches_played']}
Goals Scored: {stats['goals_scored']} ({stats['goals_per_game']:.2f} per game)
Goals Conceded: {stats['goals_conceded']}
Goal Difference: {stats['goal_difference']}
Clean Sheets: {stats['clean_sheets']}

Performance Metrics:
------------------
Average Possession: {stats['avg_possession']:.1f}%
Shot Accuracy: {stats['shot_accuracy']:.1f}%

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return report


if __name__ == '__main__':
    analyzer = TeamAnalyzer()
    analyzer.load_sample_data(team_name='Arsenal FC')
    print(analyzer.generate_report('Arsenal FC'))
    analyzer.plot_performance_trends('Arsenal FC')
