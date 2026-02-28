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

from logo_utils import _fetch_logo_pil


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

    ICON_SIZE    = 24
    ITEM_H       = 36
    WIN_W        = 440
    VISIBLE_ROWS = 12
    WIN_H        = ITEM_H * VISIBLE_ROWS + 90

    COLOR_NORMAL   = "white"
    COLOR_SELECTED = "#dce8f5"

    selected      = [None]
    selected_name = [team_list[0]]
    running       = [True]

    # ------------------------------------------------------------------
    # Root window
    # ------------------------------------------------------------------
    root = tk.Tk()
    root.title("Football Team Analyser — Select Team")
    root.resizable(False, False)
    root.update_idletasks()
    x = (root.winfo_screenwidth()  // 2) - (WIN_W // 2)
    y = (root.winfo_screenheight() // 2) - (WIN_H // 2)
    root.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")
    root.configure(bg="#f5f5f5")

    # ------------------------------------------------------------------
    # Placeholder icon
    # ------------------------------------------------------------------
    def _make_placeholder(size=ICON_SIZE):
        img = tk.PhotoImage(width=size, height=size)
        row = "{" + " ".join(["#dddddd"] * size) + "}"
        img.put(" ".join([row] * size))
        return img

    placeholder = _make_placeholder()

    # ------------------------------------------------------------------
    # Header label
    # ------------------------------------------------------------------
    tk.Label(
        root,
        text="Select a team:",
        font=("Helvetica", 11, "bold"),
        bg="#f5f5f5",
    ).pack(pady=(12, 6))

    # ------------------------------------------------------------------
    # Scrollable canvas
    # ------------------------------------------------------------------
    frame = tk.Frame(root, bg="#f5f5f5")
    frame.pack(fill="both", expand=True, padx=20)

    canvas = tk.Canvas(
        frame,
        bg="white",
        width=WIN_W - 56,
        height=ITEM_H * VISIBLE_ROWS,
        highlightthickness=1,
        highlightbackground="#cccccc",
    )
    scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
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

    # ------------------------------------------------------------------
    # Row selection helpers
    # ------------------------------------------------------------------
    row_frames  = {}
    icon_labels = {}

    def _set_row_color(name, color):
        rf = row_frames.get(name)
        if rf and rf.winfo_exists():
            rf.configure(bg=color)
            for child in rf.winfo_children():
                if child.winfo_exists():
                    child.configure(bg=color)

    def _select(name):
        _set_row_color(selected_name[0], COLOR_NORMAL)
        selected_name[0] = name
        _set_row_color(name, COLOR_SELECTED)

    # ------------------------------------------------------------------
    # Safe exit — stops background threads and destroys window
    # ------------------------------------------------------------------
    def on_confirm():
        running[0] = False
        selected[0] = selected_name[0]
        if root.winfo_exists():
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_confirm)

    # ------------------------------------------------------------------
    # Build rows (placeholder icons initially)
    # ------------------------------------------------------------------
    for name in team_list:
        rf = tk.Frame(inner, bg=COLOR_NORMAL, cursor="hand2")
        rf.pack(fill="x")
        row_frames[name] = rf

        tk.Frame(rf, height=1, bg="#eeeeee").pack(fill="x", side="bottom")

        icon = tk.Label(rf, image=placeholder, bg=COLOR_NORMAL, padx=6, pady=6)
        icon.image = placeholder
        icon.pack(side="left")
        icon_labels[name] = icon

        lbl = tk.Label(rf, text=name, font=("Helvetica", 11),
                       bg=COLOR_NORMAL, anchor="w", pady=6)
        lbl.pack(side="left", fill="x", expand=True)

        for widget in (rf, icon, lbl):
            widget.bind("<Button-1>",        lambda e, n=name: _select(n))
            widget.bind("<Double-Button-1>", lambda e, n=name: (_select(n), on_confirm()))

    _select(team_list[0])
    root.bind("<Return>", lambda e: on_confirm())

    # ------------------------------------------------------------------
    # Background icon loading via tkinter virtual events (no polling)
    # ------------------------------------------------------------------
    icon_queue = queue.Queue()
    photo_refs = {}   # keep PhotoImage references alive (prevent GC)

    def _worker(name, url):
        if not running[0]:
            return
        img = _fetch_logo_pil(url, size=(ICON_SIZE, ICON_SIZE)) if url else None
        if running[0]:
            icon_queue.put((name, img))
            root.event_generate("<<IconReady>>", when="tail")

    for name in team_list:
        threading.Thread(
            target=_worker,
            args=(name, logos.get(name)),
            daemon=True,
        ).start()

    def _on_icon_ready(event=None):
        """Drain the queue and apply any newly loaded icons."""
        while not icon_queue.empty():
            name, pil_img = icon_queue.get()
            if not running[0]:
                return
            if pil_img is not None:
                from PIL import ImageTk
                photo = ImageTk.PhotoImage(pil_img)
                photo_refs[name] = photo
                lbl = icon_labels.get(name)
                if lbl and lbl.winfo_exists():
                    lbl.configure(image=photo)
                    lbl.image = photo

    root.bind("<<IconReady>>", _on_icon_ready)

    # ------------------------------------------------------------------
    # Analyse button
    # ------------------------------------------------------------------
    tk.Button(
        root,
        text="Analyse",
        command=on_confirm,
        font=("Helvetica", 10, "bold"),
        bg="#1a73e8",
        fg="white",
        relief="flat",
        padx=18,
        pady=6,
        cursor="hand2",
    ).pack(pady=10)

    root.mainloop()
    return selected[0]
