import json
import os
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TeamAnalyzer:
    def __init__(self):
        """Initialize the analyzer with empty data structures."""
        self.team_data = None
        self.match_data = None

    @staticmethod
    def _default_match_data():
        """Static fallback data used when API configuration is unavailable."""
        return pd.DataFrame({
            'date': pd.to_datetime(['2024-01-15', '2024-01-22', '2024-01-29', '2024-02-05', '2024-02-12']),
            'team': ['Arsenal'] * 5,
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
        """Normalize SportsData.io responses to a list of game-like dicts."""
        if isinstance(payload, list):
            return payload

        if isinstance(payload, dict):
            for key in ('games', 'Games', 'data', 'Data', 'matches', 'Matches'):
                value = payload.get(key)
                if isinstance(value, list):
                    return value

        return []

    @staticmethod
    def _pick_value(item, *keys):
        """Return the first non-empty value from a mapping for any candidate keys."""
        for key in keys:
            value = item.get(key)
            if value is not None and value != '':
                return value
        return np.nan

    @staticmethod
    def _extract_row(game, team_name):
        """Convert one SportsData.io game payload into the local match row schema."""
        home_team = TeamAnalyzer._pick_value(game, 'HomeTeamName', 'HomeTeam', 'HomeTeamKey', 'HomeTeamCode')
        away_team = TeamAnalyzer._pick_value(game, 'AwayTeamName', 'AwayTeam', 'AwayTeamKey', 'AwayTeamCode')

        if pd.isna(home_team) or pd.isna(away_team):
            return None

        home_name = str(home_team).strip()
        away_name = str(away_team).strip()
        is_team_home = home_name.lower() == team_name.lower()
        is_team_away = away_name.lower() == team_name.lower()
        if not (is_team_home or is_team_away):
            return None

        home_score = TeamAnalyzer._pick_value(game, 'HomeTeamScore', 'HomeScore', 'ScoreHome')
        away_score = TeamAnalyzer._pick_value(game, 'AwayTeamScore', 'AwayScore', 'ScoreAway')

        return {
            'date': pd.to_datetime(TeamAnalyzer._pick_value(game, 'Day', 'DateTime', 'MatchTime', 'Date'), errors='coerce'),
            'team': team_name,
            'opponent': away_name if is_team_home else home_name,
            'venue': 'Home' if is_team_home else 'Away',
            'goals_scored': home_score if is_team_home else away_score,
            'goals_conceded': away_score if is_team_home else home_score,
            'possession': TeamAnalyzer._pick_value(game, 'Possession', 'PossessionPct'),
            'shots': TeamAnalyzer._pick_value(game, 'Shots', 'TeamShots'),
            'shots_on_target': TeamAnalyzer._pick_value(game, 'ShotsOnGoal', 'ShotsOnTarget'),
        }

    def load_sample_data(self):
        """Load match data from SportsData.io API, with fallback sample data."""
        api_key = os.getenv('SPORTSDATA_API_KEY')
        api_url = os.getenv('SPORTSDATA_MATCHES_URL')

        if api_key and api_url:
            self.load_match_data_from_api(api_url=api_url, api_key=api_key)
        else:
            self.match_data = self._default_match_data()

    def load_match_data_from_api(self, api_url, api_key, team_name='Arsenal'):
        """Load matches from a SportsData.io endpoint and replace `match_data`."""
        headers = {
            'Accept': 'application/json',
            'Ocp-Apim-Subscription-Key': api_key,
        }
        payload = self._fetch_json(api_url, headers=headers)
        games = self._normalize_games_payload(payload)

        rows = []
        for game in games:
            if not isinstance(game, dict):
                continue
            row = self._extract_row(game, team_name=team_name)
            if row:
                rows.append(row)

        if not rows:
            raise ValueError(
                'No matches parsed from SportsData.io response. '
                'Check endpoint, subscription key, and team naming.'
            )

        match_data = pd.DataFrame(rows)
        numeric_cols = ['goals_scored', 'goals_conceded', 'possession', 'shots', 'shots_on_target']
        for col in numeric_cols:
            if col in match_data.columns:
                match_data[col] = pd.to_numeric(match_data[col], errors='coerce')

        self.match_data = match_data.sort_values('date').reset_index(drop=True)

    def calculate_basic_stats(self, team_name):
        """Calculate basic team statistics."""
        team_matches = self.match_data[self.match_data['team'] == team_name]

        stats = {
            'matches_played': len(team_matches),
            'goals_scored': team_matches.get('goals_scored', pd.Series(dtype=float)).sum(min_count=1),
            'goals_conceded': team_matches.get('goals_conceded', pd.Series(dtype=float)).sum(min_count=1),
            'goal_difference': team_matches.get('goals_scored', pd.Series(dtype=float)).sum(min_count=1) - team_matches.get('goals_conceded', pd.Series(dtype=float)).sum(min_count=1),
            'avg_possession': team_matches['possession'].mean() if 'possession' in team_matches else np.nan,
            'shot_accuracy': (team_matches['shots_on_target'].sum() / team_matches['shots'].sum() * 100) if {'shots_on_target', 'shots'}.issubset(team_matches.columns) else np.nan,
            'goals_per_game': team_matches.get('goals_scored', pd.Series(dtype=float)).mean(),
            'clean_sheets': len(team_matches[team_matches.get('goals_conceded', pd.Series(dtype=float)) == 0]) if 'goals_conceded' in team_matches else np.nan,
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
    analyzer.load_sample_data()
    print(analyzer.generate_report('Arsenal'))
    analyzer.plot_performance_trends('Arsenal')
