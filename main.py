"""Entry point for the Football Team Analyser."""

import tkinter as tk

import matplotlib.pyplot as plt

from app_strings import (
    MSG_FETCH_LOGOS,
    MSG_FETCH_TEAM_LIST,
    MSG_LOGOS_FOUND,
    MSG_NO_TEAM_SELECTED,
    MSG_SELECTED_TEAM,
    MSG_TEAMS_FOUND,
)
from gui import pick_team_gui
from possession_prediction import open_possession_prediction_window
from team_analyser import TeamAnalyzer

SEASONS = (2025, 2026)


def _show_possession_prediction_launcher(analyzer, team_name, all_teams):
    """Show a small window with a button to open possession predictions."""

    def _open_window():
        opponents = [name for name in all_teams if name != team_name]
        predictions = analyzer.predict_possession_vs_opponents(
            team_name=team_name,
            opponents=opponents,
            seasons=SEASONS,
        )
        open_possession_prediction_window(team_name, predictions)

    root = tk.Tk()
    root.title('Advanced Predictions')
    root.geometry('360x140')
    root.configure(bg='#f5f5f5')

    tk.Label(
        root,
        text='Predict possession vs possible opponents',
        font=('Helvetica', 11, 'bold'),
        bg='#f5f5f5',
    ).pack(pady=(20, 10))

    tk.Button(
        root,
        text='Open Possession Predictor',
        command=_open_window,
        font=('Helvetica', 10, 'bold'),
        bg='#1a73e8',
        fg='white',
        relief='flat',
        padx=18,
        pady=6,
        cursor='hand2',
    ).pack()

    tk.Button(root, text='Close', command=root.destroy).pack(pady=(10, 0))
    root.mainloop()


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

    _show_possession_prediction_launcher(analyzer, team_name, all_teams)

    logo_url = logos.get(team_name)
    analyzer.plot_performance_trends(team_name, seasons=SEASONS, logo_url=logo_url)
    plt.show()


if __name__ == "__main__":
    main()
