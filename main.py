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
from app_strings import (
    MSG_FETCH_TEAM_LIST,
    MSG_TEAMS_FOUND,
    MSG_FETCH_LOGOS,
    MSG_LOGOS_FOUND,
    MSG_NO_TEAM_SELECTED,
    MSG_SELECTED_TEAM,
)

SEASONS = (2025, 2026)


def main():
    """Run the full team selection and analysis workflow."""
    analyzer = TeamAnalyzer()

    print(MSG_FETCH_TEAM_LIST)
    all_teams = analyzer.fetch_all_teams(seasons=(2026,))
    print(MSG_TEAMS_FOUND.format(count=len(all_teams)))

    print(MSG_FETCH_LOGOS)
    logos = analyzer.fetch_competition_details()
    print(MSG_LOGOS_FOUND.format(count=sum(1 for v in logos.values() if v)))

    team_name = pick_team_gui(all_teams, logos=logos)

    if not team_name:
        print(MSG_NO_TEAM_SELECTED)
        return

    print(MSG_SELECTED_TEAM.format(team_name=team_name))

    analyzer.load_sample_data(seasons=SEASONS, team_name=team_name)
    print(analyzer.generate_report(team_name, seasons=SEASONS))

    logo_url = logos.get(team_name)
    analyzer.plot_performance_trends(team_name, seasons=SEASONS, logo_url=logo_url)
    plt.show()


if __name__ == "__main__":
    main()
