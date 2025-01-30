import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class TeamAnalyzer:
    def __init__(self):
        """Initialize the analyzer with empty data structures"""
        self.team_data = None
        self.match_data = None

    def load_sample_data(self):
        """Load sample match data"""
        # Sample data structure
        self.match_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10),
            'team': ['Arsenal'] * 10,
            'opponent': ['Chelsea', 'Liverpool', 'Man City', 'Tottenham',
                         'Man United', 'Newcastle', 'Brighton', 'West Ham',
                         'Crystal Palace', 'Wolves'],
            'goals_scored': [2, 3, 1, 2, 2, 0, 3, 1, 4, 2],
            'goals_conceded': [0, 1, 2, 0, 1, 2, 0, 1, 1, 0],
            'shots': [15, 18, 12, 14, 16, 10, 20, 13, 22, 15],
            'shots_on_target': [8, 10, 5, 7, 9, 4, 12, 6, 14, 8],
            'possession': [65, 58, 45, 60, 55, 48, 62, 57, 64, 59]
        })

    def calculate_basic_stats(self, team_name):
        """Calculate basic team statistics"""
        team_matches = self.match_data[self.match_data['team'] == team_name]

        stats = {
            'matches_played': len(team_matches),
            'goals_scored': team_matches['goals_scored'].sum(),
            'goals_conceded': team_matches['goals_conceded'].sum(),
            'goal_difference': team_matches['goals_scored'].sum() - team_matches['goals_conceded'].sum(),
            'avg_possession': team_matches['possession'].mean(),
            'shot_accuracy': (team_matches['shots_on_target'].sum() / team_matches['shots'].sum() * 100),
            'goals_per_game': team_matches['goals_scored'].mean(),
            'clean_sheets': len(team_matches[team_matches['goals_conceded'] == 0])
        }

        return pd.Series(stats)

    def plot_performance_trends(self, team_name):
        """Plot performance trends over time"""
        team_matches = self.match_data[self.match_data['team'] == team_name]

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot goals
        ax1.plot(team_matches['date'], team_matches['goals_scored'],
                 label='Goals Scored', marker='o')
        ax1.plot(team_matches['date'], team_matches['goals_conceded'],
                 label='Goals Conceded', marker='o')
        ax1.set_title(f'{team_name} Goal Performance Over Time')
        ax1.legend()
        ax1.grid(True)

        # Plot possession
        ax2.plot(team_matches['date'], team_matches['possession'],
                 label='Possession %', marker='o', color='green')
        ax2.set_title(f'{team_name} Possession Over Time')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        return fig

    def generate_report(self, team_name):
        """Generate a comprehensive performance report"""
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


# Example usage
analyzer = TeamAnalyzer()
analyzer.load_sample_data()
print(analyzer.generate_report("Arsenal"))
analyzer.plot_performance_trends("Arsenal")