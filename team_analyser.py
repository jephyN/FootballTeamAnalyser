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
    # Base URL template — season year is substituted automatically.
    # Format: /TeamSeasonStats/{competition_id}/{season_year}
    # Competition ID 3 = UEFA Champions League. Adjust if needed.
    API_URL_TEMPLATE = 'https://api.sportsdata.io/v4/soccer/scores/json/TeamSeasonStats/3/{season}'
    DEFAULT_SEASONS = (2025, 2026)
    DEFAULT_TEAM_NAME = 'Arsenal FC'

    # DEFAULT_API_KEY has been removed. Set SPORTSDATA_API_KEY in your .env file.

    def __init__(self):
        """Initialize the analyzer with empty data structures."""
        self.team_data = None
        # match_data now holds rows across all loaded seasons, distinguished by
        # the 'season_year' column so per-season slicing is straightforward.
        self.match_data = None

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt(value, spec='.2f'):
        """Format a value using the given format spec, or return 'N/A' if NaN/None."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 'N/A'
        try:
            return format(value, spec)
        except (TypeError, ValueError):
            return 'N/A'

    # ------------------------------------------------------------------
    # Fallback / static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_match_data(team_name='Arsenal FC'):
        """Static fallback data used when API configuration is unavailable.

        Returns one synthetic row per season so multi-season methods still work.
        """
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
        """Fetch JSON payload from an HTTP endpoint."""
        if params:
            separator = '&' if '?' in url else '?'
            url = f"{url}{separator}{urlencode(params)}"

        req = Request(url, headers=headers or {})
        with urlopen(req, timeout=30) as response:
            payload = response.read().decode('utf-8', errors='ignore')
        return json.loads(payload)

    @staticmethod
    def _normalize_team_season_payload(payload):
        """Normalize TeamSeasonStats/Round payloads to a flat TeamSeason row list."""

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
        cleaned_url = urlunparse((
            parsed.scheme, parsed.netloc, parsed.path,
            parsed.params, cleaned_query, parsed.fragment,
        ))
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
        """Case-insensitive exact match between a row's team name and the requested name."""
        if pd.isna(row_team_name):
            return False
        return str(row_team_name).strip().lower() == str(requested_team_name).strip().lower()

    @staticmethod
    def _extract_team_season_row(team_stats, team_name='Arsenal FC'):
        """Convert a TeamSeasonStats dict into one season-aggregate row."""
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

    # ------------------------------------------------------------------
    # URL derivation
    # ------------------------------------------------------------------

    @classmethod
    def _url_for_season(cls, season_year):
        """Derive the SportsData.io TeamSeasonStats URL for a given season year."""
        return cls.API_URL_TEMPLATE.format(season=season_year)

    # ------------------------------------------------------------------
    # Raw data logging
    # ------------------------------------------------------------------

    @staticmethod
    def _log_raw_api_data(season_year, raw_rows, team_name, log_path=None):
        """Write all raw team rows for a season to a JSON log file.

        The log file is written next to the script (or to `log_path` if given).
        Each run overwrites the file entirely. Entries are keyed by season year
        and timestamped so it is clear when the data was fetched.

        Args:
            season_year:  The season year the rows belong to.
            raw_rows:     List of raw dict rows as returned by the API for the team.
            team_name:    The team name used for this fetch.
            log_path:     Optional explicit path. Defaults to team_api_data.json
                          in the same directory as this script.
        """
        if log_path is None:
            safe_name = team_name.lower().replace(' ', '_')
            log_path = Path(__file__).parent / f'{safe_name}_api_data.json'
        else:
            log_path = Path(log_path)

        # Load whatever is already in the file so we can merge season entries
        # within a single run (multiple seasons appended before the next run
        # overwrites). Because each run starts fresh, we track in-memory state
        # via a module-level dict that is reset on import.
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

    # ------------------------------------------------------------------
    # Data loading — single season (returns DataFrame, no side effects)
    # ------------------------------------------------------------------

    def load_match_data_from_api(self, api_url, api_key, team_name=DEFAULT_TEAM_NAME):
        """Fetch and parse TeamSeason rows for one season URL.

        Returns a DataFrame (does NOT update self.match_data directly).
        """
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

        # Collect raw team rows (pre-transformation) for the log
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

        # Log all raw team rows for this season
        season_year = rows[0].get('season_year', 'unknown')
        self._log_raw_api_data(season_year, raw_team_rows, team_name=team_name)

        df = pd.DataFrame(rows)

        # Only coerce genuinely numeric columns; season_type and team_id are identifiers.
        numeric_cols = [
            'goals_scored', 'goals_conceded', 'possession',
            'shots', 'shots_on_target', 'matches_in_sample', 'clean_sheets',
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    # ------------------------------------------------------------------
    # Data loading — multi-season entry point
    # ------------------------------------------------------------------

    def load_sample_data(
        self,
        api_key=None,
        seasons=DEFAULT_SEASONS,
        team_name=DEFAULT_TEAM_NAME,
    ):
        """Load data for each season in `seasons` and combine into self.match_data.

        For each season the URL is derived automatically from API_URL_TEMPLATE.
        Seasons that fail to load are skipped with a warning; if all seasons fail
        the fallback sample data is used instead.
        """
        self._load_env_file()
        # Reset the log buffer so each run overwrites the previous log file
        TeamAnalyzer._log_buffer = {}
        resolved_api_key = api_key or os.getenv('SPORTSDATA_API_KEY')

        if not resolved_api_key:
            warnings.warn(
                'SPORTSDATA_API_KEY is not configured. Using fallback sample data.',
                RuntimeWarning,
            )
            self.match_data = self._default_match_data(team_name=team_name)
            return

        season_frames = []
        for season_year in seasons:
            url = self._url_for_season(season_year)
            try:
                df = self.load_match_data_from_api(
                    api_url=url, api_key=resolved_api_key, team_name=team_name
                )
                season_frames.append(df)
            except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                warnings.warn(
                    f'Season {season_year}: failed to load ({exc}). Skipping.',
                    RuntimeWarning,
                )

        if not season_frames:
            warnings.warn(
                'All seasons failed to load. Using fallback sample data.',
                RuntimeWarning,
            )
            self.match_data = self._default_match_data(team_name=team_name)
            return

        combined = pd.concat(season_frames, ignore_index=True)
        self.match_data = combined.sort_values('date').reset_index(drop=True)

    # ------------------------------------------------------------------
    # Stats calculation — optionally filtered by season
    # ------------------------------------------------------------------

    def calculate_basic_stats(self, team_name, season_year=None):
        """Return a pandas Series of per-round average performance metrics.

        Each row in match_data represents one round. Stats are calculated as:
          - avg_possession : mean of all round possession values
          - shot_accuracy  : mean of per-round shot accuracy
                             (shots_on_target / shots * 100 per round, then averaged)

        If season_year is provided, only rows for that season are used.
        """
        mask = self.match_data['team'] == team_name
        if season_year is not None and 'season_year' in self.match_data.columns:
            mask = mask & (self.match_data['season_year'] == season_year)
        rounds = self.match_data[mask]

        num_rounds = len(rounds)

        # Average possession: sum of all round possession values / number of rounds
        if 'possession' in rounds.columns and rounds['possession'].notna().any():
            avg_possession = rounds['possession'].sum() / num_rounds
        else:
            avg_possession = np.nan

        # Shot accuracy: compute per-round accuracy, then average across rounds
        if {'shots_on_target', 'shots'}.issubset(rounds.columns):
            valid = rounds[rounds['shots'].notna() & (rounds['shots'] > 0)].copy()
            if not valid.empty:
                valid['round_accuracy'] = valid['shots_on_target'] / valid['shots'] * 100
                shot_accuracy = valid['round_accuracy'].sum() / num_rounds
            else:
                shot_accuracy = np.nan
        else:
            shot_accuracy = np.nan

        return pd.Series({
            'avg_possession': round(avg_possession, 2) if not pd.isna(avg_possession) else np.nan,
            'shot_accuracy': round(shot_accuracy, 2) if not pd.isna(shot_accuracy) else np.nan,
        }, dtype='object')

    # ------------------------------------------------------------------
    # Team discovery
    # ------------------------------------------------------------------

    def fetch_all_teams(self, api_key=None, seasons=DEFAULT_SEASONS):
        """Fetch raw payloads for each season and return a sorted deduplicated
        list of all team names found across all seasons.

        Does not modify self.match_data. Used to populate the team picker UI.
        """
        self._load_env_file()
        resolved_api_key = api_key or os.getenv('SPORTSDATA_API_KEY')

        if not resolved_api_key:
            warnings.warn(
                'SPORTSDATA_API_KEY is not configured. Returning fallback team list.',
                RuntimeWarning,
            )
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
                warnings.warn(
                    f'Season {season_year}: could not fetch team list ({exc}). Skipping.',
                    RuntimeWarning,
                )

        if not team_names:
            warnings.warn(
                'No teams found from API. Returning fallback team list.',
                RuntimeWarning,
            )
            return [self.DEFAULT_TEAM_NAME]

        return sorted(team_names)

    # ------------------------------------------------------------------
    # Multi-season side-by-side report
    # ------------------------------------------------------------------

    def generate_report(self, team_name, seasons=DEFAULT_SEASONS):
        """Generate a side-by-side performance metrics report across multiple seasons.

        Metrics are averages across all rounds in each season:
          - Avg Possession : mean possession % per round
          - Shot Accuracy  : mean per-round shot accuracy %
        """
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

    # ------------------------------------------------------------------
    # Multi-season trend chart
    # ------------------------------------------------------------------

    def plot_performance_trends(self, team_name, seasons=DEFAULT_SEASONS):
        """Plot per-round average performance metrics for each season.

        Three subplots on one figure:
          1. Goals Scored vs Conceded (grouped bars) — raw round values summed
          2. Avg Possession % per round (line)
          3. Avg Shot Accuracy % per round (line)
        """
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

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle(f'{team_name} — Season-by-Season Performance Metrics', fontsize=14, fontweight='bold')

        # --- Possession line chart ---
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

        # --- Shot accuracy line chart ---
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

        plt.tight_layout()
        return fig


# ----------------------------------------------------------------------
# Team picker GUI
# ----------------------------------------------------------------------

def pick_team_gui(team_list):
    """Open a tkinter window with a dropdown of team names.

    Blocks until the user clicks Analyse or closes the window.
    Returns the selected team name, or None if closed without confirming.
    """
    import tkinter as tk
    from tkinter import ttk

    selected = [None]

    root = tk.Tk()
    root.title("Football Team Analyser")
    root.resizable(False, False)

    root.update_idletasks()
    w, h = 360, 140
    x = (root.winfo_screenwidth() // 2) - (w // 2)
    y = (root.winfo_screenheight() // 2) - (h // 2)
    root.geometry(f"{w}x{h}+{x}+{y}")

    tk.Label(root, text="Select a team:", font=("Helvetica", 11)).pack(pady=(18, 6))

    combo_var = tk.StringVar(value=team_list[0])
    combo = ttk.Combobox(
        root,
        textvariable=combo_var,
        values=team_list,
        state="readonly",
        width=38,
        font=("Helvetica", 10),
    )
    combo.pack(pady=4)

    def on_confirm():
        selected[0] = combo_var.get()
        root.destroy()

    tk.Button(
        root,
        text="Analyse",
        command=on_confirm,
        font=("Helvetica", 10, "bold"),
        padx=12,
        pady=4,
    ).pack(pady=(10, 0))

    root.mainloop()
    return selected[0]


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    SEASONS = (2025, 2026)

    analyzer = TeamAnalyzer()

    # Step 1: fetch all unique team names across all seasons
    print("Fetching team list from API...")
    all_teams = analyzer.fetch_all_teams(seasons=SEASONS)
    print(f"{len(all_teams)} teams found.")

    # Step 2: show dropdown window, block until selection
    team_name = pick_team_gui(all_teams)

    if not team_name:
        print("No team selected. Exiting.")
    else:
        print(f"Selected: {team_name}")

        # Step 3: load data for the selected team and run the analysis
        analyzer.load_sample_data(seasons=SEASONS, team_name=team_name)

        print(analyzer.generate_report(team_name, seasons=SEASONS))

        fig = analyzer.plot_performance_trends(team_name, seasons=SEASONS)
        plt.show()