"""
Team Analyzer

Desktop analytics tool that:
- Fetches football team season data from API
- Normalizes payload variations
- Computes statistics
- Visualizes trends
- Provides Tkinter GUI team selection

Designed as a lightweight analytical client application.
"""


import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TeamAnalyzer:
    """Handles data acquisition, normalization, analytics, and reporting."""
    API_URL_TEMPLATE = 'https://api.sportsdata.io/v4/soccer/scores/json/TeamSeasonStats/3/{season}'
    DEFAULT_SEASONS = (2025, 2026)
    DEFAULT_TEAM_NAME = 'Arsenal FC'

    def __init__(self):
        self.team_data = None
        self.match_data = None

    @staticmethod
    def _fmt(value, spec='.2f'):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 'N/A'
        try:
            return format(value, spec)
        except (TypeError, ValueError):
            return 'N/A'

    @staticmethod
    def _default_match_data(team_name='Arsenal FC'):
        return pd.DataFrame({
            'date': pd.to_datetime(['2025-12-31', '2026-12-31']),
            'season_year': [2025, 2026],
            'team': [team_name] * 2,
            'possession': [56.8, 57.5],
            'shots': [495, 518],
            'shots_on_target': [172, 189],
        })

    @staticmethod
    def _fetch_json(url, headers=None, params=None):
        if params:
            separator = '&' if '?' in url else '?'
            url = f"{url}{separator}{urlencode(params)}"
        req = Request(url, headers=headers or {})
        with urlopen(req, timeout=30) as response:
            payload = response.read().decode('utf-8', errors='ignore')
        return json.loads(payload)

    @staticmethod
    def _normalize_team_season_payload(payload):
        def _coerce_team_seasons(items, round_context=None):
            rows = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                team_seasons = item.get('TeamSeasons')
                if isinstance(team_seasons, list):
                    context = {
                        'Season': item.get('Season'),
                        'SeasonType': item.get('SeasonType'),
                        'RoundId': item.get('RoundId'),
                        'RoundName': item.get('Name'),
                    }
                    for team_row in team_seasons:
                        if not isinstance(team_row, dict):
                            continue
                        merged = dict(team_row)
                        for key, value in context.items():
                            if merged.get(key) in (None, '') and value not in (None, ''):
                                merged[key] = value
                        rows.append(merged)
                else:
                    merged = dict(item)
                    if round_context:
                        for key, value in round_context.items():
                            if merged.get(key) in (None, '') and value not in (None, ''):
                                merged[key] = value
                    rows.append(merged)
            return rows

        if isinstance(payload, list):
            return _coerce_team_seasons(payload)
        if isinstance(payload, dict):
            for key in ('TeamSeasonStats', 'teamSeasonStats', 'data', 'Data'):
                value = payload.get(key)
                if isinstance(value, list):
                    return _coerce_team_seasons(value)
                if isinstance(value, dict):
                    return _coerce_team_seasons([value])
            for key in ('Rounds', 'rounds'):
                rounds = payload.get(key)
                if isinstance(rounds, list):
                    return _coerce_team_seasons(rounds)
            if isinstance(payload.get('TeamSeasons'), list):
                context = {
                    'Season': payload.get('Season'),
                    'SeasonType': payload.get('SeasonType'),
                    'RoundId': payload.get('RoundId'),
                    'RoundName': payload.get('Name'),
                }
                return _coerce_team_seasons(payload.get('TeamSeasons'), round_context=context)
            return _coerce_team_seasons([payload])
        return []

    @staticmethod
    def _load_env_file(env_path='.env'):
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
        if not url:
            return url, None
        parsed = urlparse(url)
        query = parse_qs(parsed.query, keep_blank_values=True)
        values = query.pop('key', None)
        extracted_key = values[0] if values else None
        cleaned_query = urlencode(query, doseq=True)
        cleaned_url = urlunparse((
            parsed.scheme, parsed.netloc, parsed.path,
            parsed.params, cleaned_query, parsed.fragment,
        ))
        return cleaned_url, extracted_key

    @staticmethod
    def _pick_value(item, *keys):
        for key in keys:
            value = item.get(key)
            if value is not None and value != '':
                return value
        return np.nan

    @staticmethod
    def _team_matches(row_team_name, requested_team_name):
        if pd.isna(row_team_name):
            return False
        return str(row_team_name).strip().lower() == str(requested_team_name).strip().lower()

    @staticmethod
    def _extract_team_season_row(team_stats, team_name='Arsenal FC'):
        source_team = TeamAnalyzer._pick_value(team_stats, 'Name', 'TeamName', 'Team', 'TeamKey')
        if not TeamAnalyzer._team_matches(source_team, team_name):
            return None
        season = TeamAnalyzer._pick_value(team_stats, 'Season', 'SeasonYear')
        season_year = pd.to_numeric(season, errors='coerce')
        season_date = (
            pd.to_datetime(f"{int(season_year)}-12-31", errors='coerce')
            if not pd.isna(season_year)
            else pd.NaT
        )
        clean_sheets = TeamAnalyzer._pick_value(
            team_stats, 'GoalkeeperCleanSheets', 'DefenderCleanSheets', 'CleanSheets'
        )
        return {
            'date': season_date,
            'season_year': int(season_year) if not pd.isna(season_year) else np.nan,
            'team': team_name,
            'opponent': 'All Teams (Season Aggregate)',
            'venue': 'Season',
            'goals_scored': TeamAnalyzer._pick_value(team_stats, 'Score', 'Goals', 'GoalsScored', 'TeamGoals'),
            'goals_conceded': TeamAnalyzer._pick_value(team_stats, 'OpponentScore', 'OpponentGoals', 'GoalsAgainst', 'GoalsConceded'),
            'possession': TeamAnalyzer._pick_value(team_stats, 'Possession', 'PossessionPct', 'PossessionPercentage'),
            'shots': TeamAnalyzer._pick_value(team_stats, 'Shots', 'ShotsTotal', 'TeamShots'),
            'shots_on_target': TeamAnalyzer._pick_value(team_stats, 'ShotsOnGoal', 'ShotsOnTarget'),
            'matches_in_sample': TeamAnalyzer._pick_value(team_stats, 'Games', 'GamesPlayed', 'Matches', 'MatchesPlayed'),
            'clean_sheets': clean_sheets,
            'season_type': TeamAnalyzer._pick_value(team_stats, 'SeasonType'),
            'team_id': TeamAnalyzer._pick_value(team_stats, 'TeamId'),
        }

    @classmethod
    def _url_for_season(cls, season_year):
        return cls.API_URL_TEMPLATE.format(season=season_year)

    @staticmethod
    def _log_raw_api_data(season_year, raw_rows, team_name, log_path=None):
        if log_path is None:
            safe_name = team_name.lower().replace(' ', '_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = Path(__file__).parent / f'{safe_name}_api_data_{timestamp}.json'
        else:
            log_path = Path(log_path)
        existing = getattr(TeamAnalyzer, '_log_buffer', {})
        existing[str(season_year)] = {
            'fetched_at': datetime.now().isoformat(timespec='seconds'),
            'season_year': season_year,
            'team': team_name,
            'row_count': len(raw_rows),
            'data': raw_rows,
        }
        TeamAnalyzer._log_buffer = existing
        log_path.write_text(
            json.dumps({'arsenal_api_log': existing}, indent=2, default=str),
            encoding='utf-8',
        )

    def load_match_data_from_api(self, api_url, api_key, team_name=DEFAULT_TEAM_NAME):
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
        normalized = self._normalize_team_season_payload(payload)
        raw_team_rows = [
            item for item in normalized
            if isinstance(item, dict)
            and self._team_matches(
                self._pick_value(item, 'Name', 'TeamName', 'Team', 'TeamKey'),
                team_name,
            )
        ]
        for season_row in normalized:
            row = self._extract_team_season_row(season_row, team_name=team_name)
            if row:
                rows.append(row)
        if not rows:
            raise ValueError(f'No {team_name} data parsed from: {cleaned_url}')
        season_year = rows[0].get('season_year', 'unknown')
        self._log_raw_api_data(season_year, raw_team_rows, team_name=team_name)
        df = pd.DataFrame(rows)
        numeric_cols = [
            'goals_scored', 'goals_conceded', 'possession',
            'shots', 'shots_on_target', 'matches_in_sample', 'clean_sheets',
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def load_sample_data(self, api_key=None, seasons=DEFAULT_SEASONS, team_name=DEFAULT_TEAM_NAME):
        self._load_env_file()
        TeamAnalyzer._log_buffer = {}
        resolved_api_key = api_key or os.getenv('SPORTSDATA_API_KEY')
        if not resolved_api_key:
            warnings.warn('SPORTSDATA_API_KEY is not configured. Using fallback sample data.', RuntimeWarning)
            self.match_data = self._default_match_data(team_name=team_name)
            return
        season_frames = []
        for season_year in seasons:
            url = self._url_for_season(season_year)
            try:
                df = self.load_match_data_from_api(api_url=url, api_key=resolved_api_key, team_name=team_name)
                season_frames.append(df)
            except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                warnings.warn(f'Season {season_year}: failed to load ({exc}). Skipping.', RuntimeWarning)
        if not season_frames:
            warnings.warn('All seasons failed to load. Using fallback sample data.', RuntimeWarning)
            self.match_data = self._default_match_data(team_name=team_name)
            return
        combined = pd.concat(season_frames, ignore_index=True)
        self.match_data = combined.sort_values('date').reset_index(drop=True)

    def calculate_basic_stats(self, team_name, season_year=None):
        mask = self.match_data['team'] == team_name
        if season_year is not None and 'season_year' in self.match_data.columns:
            mask = mask & (self.match_data['season_year'] == season_year)
        rounds = self.match_data[mask]
        if 'possession' in rounds.columns and rounds['possession'].notna().any():
            avg_possession = rounds['possession'].mean()
        else:
            avg_possession = np.nan
        if {'shots_on_target', 'shots'}.issubset(rounds.columns):
            valid = rounds[rounds['shots'].notna() & (rounds['shots'] > 0)].copy()
            if not valid.empty:
                valid['round_accuracy'] = valid['shots_on_target'] / valid['shots'] * 100
                shot_accuracy = valid['round_accuracy'].mean()
            else:
                shot_accuracy = np.nan
        else:
            shot_accuracy = np.nan
        return pd.Series({
            'avg_possession': round(avg_possession, 2) if not pd.isna(avg_possession) else np.nan,
            'shot_accuracy': round(shot_accuracy, 2) if not pd.isna(shot_accuracy) else np.nan,
        }, dtype='object')

    COMPETITION_DETAILS_TEMPLATE = (
        'https://api.sportsdata.io/v4/soccer/scores/json/CompetitionDetails/{competition_id}'
    )

    def fetch_competition_details(self, api_key=None, competition_id=3):
        self._load_env_file()
        resolved_api_key = api_key or os.getenv('SPORTSDATA_API_KEY')
        if not resolved_api_key:
            warnings.warn('SPORTSDATA_API_KEY is not configured. Cannot fetch competition details.', RuntimeWarning)
            return {}
        url = self.COMPETITION_DETAILS_TEMPLATE.format(competition_id=competition_id)
        headers = {
            'Accept': 'application/json',
            'Ocp-Apim-Subscription-Key': resolved_api_key,
        }
        try:
            payload = self._fetch_json(url, headers=headers)
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            warnings.warn(f'Failed to fetch competition details ({exc}). No logos will be shown.', RuntimeWarning)
            return {}
        logos = {}
        teams = payload.get('Teams') or payload.get('teams') or []
        for team in teams:
            if not isinstance(team, dict):
                continue
            name = (
                team.get('Name')
                or team.get('TeamName')
                or team.get('ShortName')
            )
            logo_url = team.get('WikipediaLogoUrl') or team.get('wikipediaLogoUrl')
            if name:
                logos[str(name).strip()] = logo_url or None
        return logos

    def fetch_all_teams(self, api_key=None, seasons=DEFAULT_SEASONS):
        self._load_env_file()
        resolved_api_key = api_key or os.getenv('SPORTSDATA_API_KEY')
        if not resolved_api_key:
            warnings.warn('SPORTSDATA_API_KEY is not configured. Returning fallback team list.', RuntimeWarning)
            return [self.DEFAULT_TEAM_NAME]
        team_names = set()
        for season_year in seasons:
            url = self._url_for_season(season_year)
            cleaned_url, key_from_url = self._extract_api_key_from_url(url)
            resolved_key = resolved_api_key or key_from_url
            headers = {
                'Accept': 'application/json',
                'Ocp-Apim-Subscription-Key': resolved_key,
            }
            try:
                payload = self._fetch_json(cleaned_url, headers=headers)
                rows = self._normalize_team_season_payload(payload)
                for item in rows:
                    if not isinstance(item, dict):
                        continue
                    name = self._pick_value(item, 'Name', 'TeamName', 'Team', 'TeamKey')
                    if not (name is None or (isinstance(name, float) and np.isnan(name))):
                        team_names.add(str(name).strip())
            except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                warnings.warn(f'Season {season_year}: could not fetch team list ({exc}). Skipping.', RuntimeWarning)
        if not team_names:
            warnings.warn('No teams found from API. Returning fallback team list.', RuntimeWarning)
            return [self.DEFAULT_TEAM_NAME]
        return sorted(team_names)

    def generate_report(self, team_name, seasons=DEFAULT_SEASONS):
        season_stats = {}
        available_seasons = []
        for yr in seasons:
            if 'season_year' in self.match_data.columns and yr not in self.match_data['season_year'].values:
                continue
            season_stats[yr] = self.calculate_basic_stats(team_name, season_year=yr)
            available_seasons.append(yr)
        if not available_seasons:
            return f'No data available for {team_name} across seasons {seasons}.'
        label_w = 24
        col_w = 14
        header_seasons = ''.join(str(yr).rjust(col_w) for yr in available_seasons)
        separator = '-' * (label_w + col_w * len(available_seasons))

        def row(label, key, spec='.2f', suffix=''):
            values = ''
            for yr in available_seasons:
                raw = season_stats[yr].get(key, np.nan)
                cell = self._fmt(raw, spec) + suffix if not (raw is None or (isinstance(raw, float) and np.isnan(raw))) else 'N/A'
                values += cell.rjust(col_w)
            return f'{label:<{label_w}}{values}'

        lines = [
            f'\nPerformance Report — {team_name}',
            f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            f'(Values are averages per round across each season)',
            '',
            f'{"Season":<{label_w}}{header_seasons}',
            separator,
            'Performance Metrics:',
            row('  Avg Possession', 'avg_possession', '.2f', '%'),
            row('  Shot Accuracy', 'shot_accuracy', '.2f', '%'),
            separator,
        ]
        return '\n'.join(lines)

    def plot_performance_trends(self, team_name, seasons=DEFAULT_SEASONS, logo_url=None):
        available_seasons = []
        possession_vals = []
        shot_accuracy_vals = []

        for yr in seasons:
            if 'season_year' in self.match_data.columns and yr not in self.match_data['season_year'].values:
                continue
            s = self.calculate_basic_stats(team_name, season_year=yr)
            available_seasons.append(yr)
            possession_vals.append(s.get('avg_possession', np.nan))
            shot_accuracy_vals.append(s.get('shot_accuracy', np.nan))

        season_labels = [str(yr) for yr in available_seasons]

        def _is_nan(v):
            return v is None or (isinstance(v, float) and np.isnan(v))

        def _annotate_line(ax, labels, values, fmt='.2f', suffix='%'):
            for lbl, val in zip(labels, values):
                if not _is_nan(val):
                    ax.annotate(
                        f'{val:{fmt}}{suffix}',
                        (lbl, val),
                        textcoords='offset points',
                        xytext=(0, 8),
                        ha='center',
                        fontsize=9,
                    )

        logo_img = _fetch_logo_pil(logo_url, size=(52, 52)) if logo_url else None
        has_logo = logo_img is not None

        TOP = 0.86
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        has_possession = any(not _is_nan(v) for v in possession_vals)
        if has_possession:
            ax1.plot(season_labels, possession_vals, marker='o', color='green',
                     linewidth=2, markersize=8, label='Avg Possession %')
            _annotate_line(ax1, season_labels, possession_vals)
            ax1.set_title('Average Possession % per Round')
            ax1.set_ylabel('Possession (%)')
            ax1.set_ylim(0, 100)
        else:
            ax1.text(0.5, 0.5, 'Possession data not available', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_axis_off()
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.6)

        has_accuracy = any(not _is_nan(v) for v in shot_accuracy_vals)
        if has_accuracy:
            ax2.plot(season_labels, shot_accuracy_vals, marker='s', color='darkorange',
                     linewidth=2, markersize=8, label='Avg Shot Accuracy %')
            _annotate_line(ax2, season_labels, shot_accuracy_vals)
            ax2.set_title('Average Shot Accuracy % per Round')
            ax2.set_ylabel('Shot Accuracy (%)')
            ax2.set_ylim(0, 100)
        else:
            ax2.text(0.5, 0.5, 'Shot accuracy data not available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_axis_off()
        ax2.legend()
        ax2.grid(axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, TOP])

        if has_logo:
            logo_ax = fig.add_axes([0.01, TOP + 0.005, 0.08, 1.0 - TOP - 0.01])
            logo_ax.imshow(np.array(logo_img))
            logo_ax.axis('off')
            title_x = 0.54
        else:
            title_x = 0.5

        fig.suptitle(
            f'{team_name} — Season-by-Season Performance Metrics',
            fontsize=14, fontweight='bold',
            y=(TOP + 1.0) / 2,
            x=title_x,
        )

        return fig


# ----------------------------------------------------------------------
# Logo helpers
# ----------------------------------------------------------------------

def _wikimedia_thumbnail_url(upload_url, width=320):
    import re
    if not upload_url:
        return upload_url
    m = re.match(
        r"https://upload\.wikimedia\.org/wikipedia/([^/]+)/(?:thumb/)?[a-f0-9]/[a-f0-9]{2}/([^/]+?)(?:/\d+px-.+)?$",
        upload_url,
    )
    if not m:
        return upload_url
    wiki, filename = m.group(1), m.group(2)
    host = "commons.wikimedia.org" if wiki == "commons" else f"{wiki}.wikipedia.org"
    return f"https://{host}/w/thumb.php?f={filename}&w={width}"


def _fetch_logo_pil(url, size=(80, 80)):
    import io
    import time
    try:
        from PIL import Image
    except ImportError:
        return None
    thumb_url = _wikimedia_thumbnail_url(url, width=320)
    headers = {
        "User-Agent": "football-team-analyser/1.0 (educational project; python-urllib)",
        "Accept": "image/png,image/*",
    }
    for attempt in range(3):
        try:
            req = Request(thumb_url, headers=headers)
            with urlopen(req, timeout=15) as resp:
                data = resp.read()
            return Image.open(io.BytesIO(data)).convert("RGBA").resize(size, Image.LANCZOS)
        except (OSError, ValueError, RuntimeError) as exc:
            if "429" in str(exc) and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            break
    return None


def pick_team_gui(team_list, logos=None):
    import tkinter as tk
    import threading
    import queue

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

    root = tk.Tk()
    root.title("Football Team Analyser — Select Team")
    root.resizable(False, False)

    root.update_idletasks()
    x = (root.winfo_screenwidth()  // 2) - (WIN_W // 2)
    y = (root.winfo_screenheight() // 2) - (WIN_H // 2)
    root.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")
    root.configure(bg="#f5f5f5")

    # -------------------------------------------------
    # placeholder icon
    # -------------------------------------------------

    def _make_placeholder(size=ICON_SIZE):
        img = tk.PhotoImage(width=size, height=size)
        row = "{" + " ".join(["#dddddd"] * size) + "}"
        img.put(" ".join([row] * size))
        return img

    placeholder = _make_placeholder()

    tk.Label(
        root,
        text="Select a team:",
        font=("Helvetica", 11, "bold"),
        bg="#f5f5f5",
    ).pack(pady=(12, 6))

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
    canvas.bind("<Button-4>", _on_mousewheel)
    canvas.bind("<Button-5>", _on_mousewheel)

    row_frames  = {}
    icon_labels = {}

    def _set_row_color(name, color):
        rf = row_frames.get(name)
        if rf and rf.winfo_exists():
            rf.configure(bg=color)
            for c in rf.winfo_children():
                if c.winfo_exists():
                    c.configure(bg=color)

    def _select(name):
        _set_row_color(selected_name[0], COLOR_NORMAL)
        selected_name[0] = name
        _set_row_color(name, COLOR_SELECTED)

    # -------------------------------------------------
    # SAFE EXIT
    # -------------------------------------------------

    def on_confirm():
        running[0] = False
        selected[0] = selected_name[0]
        if root.winfo_exists():
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_confirm)

    # -------------------------------------------------
    # rows
    # -------------------------------------------------

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

        for w in (rf, icon, lbl):
            w.bind("<Button-1>", lambda e, n=name: _select(n))
            w.bind("<Double-Button-1>", lambda e, n=name: (_select(n), on_confirm()))

    _select(team_list[0])
    root.bind("<Return>", lambda e: on_confirm())

    # -------------------------------------------------
    # THREAD → TK EVENT BRIDGE
    # -------------------------------------------------

    icon_queue = queue.Queue()
    photo_refs = {}

    def worker(name, url):
        if not running[0]:
            return
        img = _fetch_logo_pil(url, size=(ICON_SIZE, ICON_SIZE)) if url else None
        if running[0]:
            icon_queue.put((name, img))
            root.event_generate("<<IconReady>>", when="tail")

    for name in team_list:
        threading.Thread(
            target=worker,
            args=(name, logos.get(name)),
            daemon=True,
        ).start()

    # -------------------------------------------------
    # MAIN THREAD HANDLER (NO POLLING)
    # -------------------------------------------------

    def on_icon_ready(event=None):
        while not icon_queue.empty():
            name, pil_img = icon_queue.get()

            if not running[0]:
                return

            if pil_img:
                from PIL import ImageTk
                photo = ImageTk.PhotoImage(pil_img)
                photo_refs[name] = photo

                lbl = icon_labels.get(name)
                if lbl and lbl.winfo_exists():
                    lbl.configure(image=photo)
                    lbl.image = photo

    root.bind("<<IconReady>>", on_icon_ready)

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


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    SEASONS = (2025, 2026)

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
    else:
        print(f"Selected: {team_name}")

        analyzer.load_sample_data(seasons=SEASONS, team_name=team_name)

        print(analyzer.generate_report(team_name, seasons=SEASONS))

        logo_url = logos.get(team_name)
        fig = analyzer.plot_performance_trends(team_name, seasons=SEASONS, logo_url=logo_url)
        plt.show()
