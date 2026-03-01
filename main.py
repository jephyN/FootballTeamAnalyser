"""
main.py

Entry point for the Football Team Analyser.

Orchestrates the startup sequence:
  1. Fetch team list for the dropdown (2026 season only)
  2. Fetch logo URLs for all teams in the competition
  3. Show the team picker GUI
  4. Load season data for the selected team (2025 + 2026)
  5. Print the performance report
  6. Display the trend chart
"""

import matplotlib.pyplot as plt

from team_analyser import TeamAnalyzer
from gui import pick_team_gui

SEASONS = (2025, 2026)


def main():
    """Run the full team selection and analysis workflow."""
    analyzer = TeamAnalyzer()

    print("Fetching team list from API...")
    all_teams = analyzer.fetch_all_teams(seasons=(2026,))
    print(f"{len(all_teams)} teams found.")

    print("Fetching team logo URLs...")
    logos = analyzer.fetch_competition_details()
    print(f"{sum(1 for v in logos.values() if v)} logo URLs found.")

    team_name = pick_team_gui(all_teams, logos=logos)

    if not team_name:
        print("No team selected. Exiting.")
        return

    print(f"Selected: {team_name}")

    analyzer.load_sample_data(seasons=SEASONS, team_name=team_name)
    print(analyzer.generate_report(team_name, seasons=SEASONS))

    logo_url = logos.get(team_name)
    analyzer.plot_performance_trends(team_name, seasons=SEASONS, logo_url=logo_url)
    plt.show()


if __name__ == "__main__":
    main()
