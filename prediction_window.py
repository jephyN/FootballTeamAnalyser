"""GUI window for displaying predicted possessions."""

import tkinter as tk
from tkinter import messagebox


def open_possession_prediction_window(team_name, predictions):
    """Show predictions in a standalone tkinter window."""
    if predictions.empty:
        messagebox.showwarning('Possession prediction', f'No predictions available for {team_name}.')
        return

    win = tk.Tk()
    win.title(f'Predicted possession — {team_name}')
    win.geometry('620x420')
    win.configure(bg='#f5f5f5')

    tk.Label(
        win,
        text=f'Predicted possession for upcoming rounds ({team_name})',
        font=('Helvetica', 12, 'bold'),
        bg='#f5f5f5',
    ).pack(pady=(12, 8))

    text = tk.Text(win, wrap='none', font=('Courier New', 10), bg='white')
    text.pack(fill='both', expand=True, padx=12, pady=8)

    text.insert('end', f"{'Opponent':<30} {'Team %':>10} {'Opponent %':>12}\n")
    text.insert('end', '-' * 56 + '\n')
    for _, row in predictions.iterrows():
        text.insert(
            'end',
            f"{row['opponent']:<30} {row['team_possession_pred']:>10.2f} {row['opponent_possession_pred']:>12.2f}\n",
        )

    text.configure(state='disabled')
    tk.Button(
        win,
        text='Close',
        command=win.destroy,
        font=('Helvetica', 10, 'bold'),
        bg='#1a73e8',
        fg='white',
        relief='flat',
        padx=18,
        pady=6,
    ).pack(pady=(0, 12))

    win.mainloop()
