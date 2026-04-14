"""
gui.py

Tkinter team picker GUI.

Responsibilities:
- Display a scrollable list of team names with small logo icons
- Load icons in background threads; deliver to the main thread via
  tkinter's virtual event system (no polling loop)
- Return the selected team name, or None if the window is closed
"""

import queue
import threading
import tkinter as tk
from dataclasses import dataclass

from PIL import ImageTk

from logo_utils import _fetch_logo_pil


@dataclass(frozen=True)
class _GuiConfig:
    """Immutable layout and colour configuration for the team picker window."""
    icon_size: int = 24
    item_height: int = 36
    window_width: int = 440
    visible_rows: int = 12
    color_normal: str = "white"
    color_selected: str = "#dce8f5"

    @property
    def window_height(self):
        """Total window height derived from row count."""
        return self.item_height * self.visible_rows + 90


_CFG = _GuiConfig()
_STATE_SELECTED = "selected"
_STATE_SELECTED_NAME = "selected_name"
_STATE_RUNNING = "running"
_EVENT_ICON_READY = "<<IconReady>>"


def _make_placeholder():
    """Return a grey square tk.PhotoImage used as a placeholder icon."""
    size = _CFG.icon_size
    img = tk.PhotoImage(width=size, height=size)
    row = "{" + " ".join(["#dddddd"] * size) + "}"
    img.put(" ".join([row] * size))
    return img


def _build_scrollable_canvas(parent):
    """Create a scrollable canvas inside parent; return (canvas, inner_frame)."""
    canvas = tk.Canvas(
        parent,
        bg="white",
        width=_CFG.window_width - 56,
        height=_CFG.item_height * _CFG.visible_rows,
        highlightthickness=1,
        highlightbackground="#cccccc",
    )
    scrollbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    inner = tk.Frame(canvas, bg="white")
    canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))

    def _on_mousewheel(event):
        delta = -1 * (event.delta // 120) if event.delta else (-1 if event.num == 4 else 1)
        canvas.yview_scroll(delta, "units")

    canvas.bind("<MouseWheel>", _on_mousewheel)
    canvas.bind("<Button-4>",   _on_mousewheel)
    canvas.bind("<Button-5>",   _on_mousewheel)

    return canvas, inner


def _build_team_rows(inner, team_list, placeholder):
    """Populate inner frame with one row per team; return (row_frames, icon_labels)."""
    row_frames = {}
    icon_labels = {}
    for name in team_list:
        row_frame = tk.Frame(inner, bg=_CFG.color_normal, cursor="hand2")
        row_frame.pack(fill="x")
        row_frames[name] = row_frame

        tk.Frame(row_frame, height=1, bg="#eeeeee").pack(fill="x", side="bottom")

        icon = tk.Label(
            row_frame, image=placeholder, bg=_CFG.color_normal, padx=6, pady=6
        )
        icon.image = placeholder
        icon.pack(side="left")
        icon_labels[name] = icon

        tk.Label(
            row_frame, text=name, font=("Helvetica", 11),
            bg=_CFG.color_normal, anchor="w", pady=6,
        ).pack(side="left", fill="x", expand=True)

    return row_frames, icon_labels


def _start_icon_workers(root, team_list, logos, state, icon_queue):
    """Spawn one daemon thread per team to fetch logo images."""
    def _worker(team_name, url):
        if not state[_STATE_RUNNING]:
            return
        img = _fetch_logo_pil(url, size=(_CFG.icon_size, _CFG.icon_size)) if url else None
        if state[_STATE_RUNNING]:
            icon_queue.put((team_name, img))
            root.event_generate(_EVENT_ICON_READY, when="tail")

    for name in team_list:
        threading.Thread(
            target=_worker, args=(name, logos.get(name)), daemon=True
        ).start()


def _create_picker_window():
    """Create and center the picker root window."""
    root = tk.Tk()
    root.title("Football Team Analyser — Select Team")
    root.resizable(False, False)
    root.update_idletasks()
    offset_x = (root.winfo_screenwidth() // 2) - (_CFG.window_width // 2)
    offset_y = (root.winfo_screenheight() // 2) - (_CFG.window_height // 2)
    root.geometry(f"{_CFG.window_width}x{_CFG.window_height}+{offset_x}+{offset_y}")
    root.configure(bg="#f5f5f5")
    return root


def _bind_team_row_clicks(row_frames, icon_labels, team_name, select_row, on_confirm):
    """Bind single/double-click handlers for a team row and icon."""
    widgets = row_frames[team_name].winfo_children() + [icon_labels[team_name]]
    for widget in widgets:
        widget.bind("<Button-1>", lambda e, n=team_name: select_row(n))
        widget.bind("<Double-Button-1>", lambda e, n=team_name: (select_row(n), on_confirm()))


def _drain_icon_queue(icon_queue, icon_labels, photo_refs, state):
    """Drain queued logo images and apply them to row labels."""
    while not icon_queue.empty():
        name, pil_img = icon_queue.get()
        if not state[_STATE_RUNNING]:
            return
        if pil_img is not None:
            photo = ImageTk.PhotoImage(pil_img)
            photo_refs[name] = photo
            lbl_widget = icon_labels.get(name)
            if lbl_widget and lbl_widget.winfo_exists():
                lbl_widget.configure(image=photo)
                lbl_widget.image = photo


def _build_selection_handlers(root, row_frames, state):
    """Return (select_row, on_confirm) callbacks for row highlighting and close."""
    def _set_row_color(team_name, color):
        frame_widget = row_frames.get(team_name)
        if frame_widget and frame_widget.winfo_exists():
            frame_widget.configure(bg=color)
            for child in frame_widget.winfo_children():
                if child.winfo_exists():
                    child.configure(bg=color)

    def _select(team_name):
        _set_row_color(state[_STATE_SELECTED_NAME], _CFG.color_normal)
        state[_STATE_SELECTED_NAME] = team_name
        _set_row_color(team_name, _CFG.color_selected)

    def _on_confirm():
        state[_STATE_RUNNING] = False
        state[_STATE_SELECTED] = state[_STATE_SELECTED_NAME]
        if root.winfo_exists():
            root.destroy()

    return _select, _on_confirm


def _bind_icon_updates(root, icon_queue, icon_labels, photo_refs, state):
    """Bind icon-ready event to queue-draining callback."""
    root.bind(
        _EVENT_ICON_READY,
        lambda _event=None: _drain_icon_queue(icon_queue, icon_labels, photo_refs, state),
    )


def pick_team_gui(team_list, logos=None):
    """Open a tkinter window with a scrollable list of teams.

    Each row shows a 24x24 logo icon beside the team name. The window
    opens instantly with grey placeholder icons; real logos populate as
    background threads complete their downloads.

    Args:
        team_list : sorted list of team name strings.
        logos     : dict of {team_name: logo_url}. Teams with no URL
                    keep the grey placeholder.

    Returns:
        The selected team name, or None if closed without confirming.
    """
    if logos is None:
        logos = {}

    state = {
        _STATE_SELECTED: None,
        _STATE_SELECTED_NAME: team_list[0],
        _STATE_RUNNING: True,
    }

    root = _create_picker_window()

    placeholder = _make_placeholder()
    tk.Label(root, text="Select a team:", font=("Helvetica", 11, "bold"),
             bg="#f5f5f5").pack(pady=(12, 6))

    scroll_frame = tk.Frame(root, bg="#f5f5f5")
    scroll_frame.pack(fill="both", expand=True, padx=20)
    _, inner = _build_scrollable_canvas(scroll_frame)

    row_frames, icon_labels = _build_team_rows(inner, team_list, placeholder)

    _select, on_confirm = _build_selection_handlers(root, row_frames, state)

    root.protocol("WM_DELETE_WINDOW", on_confirm)

    for name in team_list:
        _bind_team_row_clicks(row_frames, icon_labels, name, _select, on_confirm)

    _select(team_list[0])
    root.bind("<Return>", lambda e: on_confirm())

    icon_queue = queue.Queue()
    photo_refs = {}

    _start_icon_workers(root, team_list, logos, state, icon_queue)
    _bind_icon_updates(root, icon_queue, icon_labels, photo_refs, state)

    tk.Button(
        root, text="Analyse", command=on_confirm,
        font=("Helvetica", 10, "bold"), bg="#1a73e8", fg="white",
        relief="flat", padx=18, pady=6, cursor="hand2",
    ).pack(pady=10)

    root.mainloop()
    return state[_STATE_SELECTED]
