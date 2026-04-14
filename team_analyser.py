"""
team_analyser.py

Handles data acquisition, payload normalisation, statistics calculation,
and report/chart generation for football team season data.
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

from logo_utils import _fetch_logo_pil

_PV = "GoalkeeperCleanSheets", "DefenderCleanSheets", "CleanSheets"

_NUMERIC_COLS = [
    'goals_scored', 'goals_conceded', 'possession',
    'shots', 'shots_on_target', 'matches_in_sample', 'clean_sheets',
]


def _coerce_numeric_cols(data_frame):
    """Coerce known numeric columns to float in-place."""
    for col in _NUMERIC_COLS:
        if col in data_frame.columns:
            data_frame[col] = pd.to_numeric(data_frame[col], errors='coerce')


def _build_api_headers(api_key):
    """Return the standard SportsData.io request headers dict."""
    return {'Accept': 'application/json', 'Ocp-Apim-Subscription-Key': api_key}


def _collect_season_stats(analyzer, team_name, seasons):
    """Return (available_seasons, possession_vals, shot_accuracy_vals) lists."""
    available_seasons, possession_vals, shot_accuracy_vals = [], [], []
    for yr in seasons:
        if (
            'season_year' in analyzer.match_data.columns
            and yr not in analyzer.match_data['season_year'].values
        ):
            continue
        stats = analyzer.calculate_basic_stats(team_name, season_year=yr)
        available_seasons.append(yr)
        possession_vals.append(stats.get('avg_possession', np.nan))
        shot_accuracy_vals.append(stats.get('shot_accuracy', np.nan))
    return available_seasons, possession_vals, shot_accuracy_vals


def _add_chart_logo(fig, logo_img, title_top):
    """Overlay the team logo in the header strip; return title x offset."""
    logo_ax = fig.add_axes([0.01, title_top + 0.005, 0.08, 1.0 - title_top - 0.01])
    logo_ax.imshow(np.array(logo_img))
    logo_ax.axis('off')
    return 0.54


def _is_nan(value):
    """Return True if value is None or a float NaN."""
    return value is None or (isinstance(value, float) and np.isnan(value))


def _build_season_row(team_stats, pick, team_name):
    """Extract and return a normalised season-row dict from a raw stats dict."""
    goals_scored = pick(team_stats, 'Score', 'Goals', 'GoalsScored', 'TeamGoals')
    goals_conceded = pick(
        team_stats, 'OpponentScore', 'OpponentGoals', 'GoalsAgainst', 'GoalsConceded'
    )
    possession = pick(team_stats, 'Possession', 'PossessionPct', 'PossessionPercentage')
    shots = pick(team_stats, 'Shots', 'ShotsTotal', 'TeamShots')
    shots_on_target = pick(team_stats, 'ShotsOnGoal', 'ShotsOnTarget')
    matches_in_sample = pick(
        team_stats, 'Games', 'GamesPlayed', 'Matches', 'MatchesPlayed'
    )
    clean_sheets = pick(team_stats, *_PV)
    season = pick(team_stats, 'Season', 'SeasonYear')
    season_year = pd.to_numeric(season, errors='coerce')
    season_date = (
        pd.to_datetime(f"{int(season_year)}-12-31", errors='coerce')
        if not pd.isna(season_year) else pd.NaT
    )
    return {
        'date': season_date,
        'season_year': int(season_year) if not pd.isna(season_year) else np.nan,
        'team': team_name,
        'opponent': 'All Teams (Season Aggregate)',
        'venue': 'Season',
        'goals_scored': goals_scored,
        'goals_conceded': goals_conceded,
        'possession': possession,
        'shots': shots,
        'shots_on_target': shots_on_target,
        'matches_in_sample': matches_in_sample,
        'clean_sheets': clean_sheets,
        'season_type': pick(team_stats, 'SeasonType'),
        'team_id': pick(team_stats, 'TeamId'),
    }


def _annotate_line(ax, labels, values, fmt='.2f', suffix='%'):
    """Annotate each point on a line chart with its formatted value."""
    for lbl, val in zip(labels, values):
        if not _is_nan(val):
            ax.annotate(
                f'{val:{fmt}}{suffix}', (lbl, val),
                textcoords='offset points', xytext=(0, 8),
                ha='center', fontsize=9,
            )


def _build_possession_training_frame(team_possession):
    """Build pairwise training rows from per-team average possession values."""
    teams = list(team_possession.items())
    rows = []
    for team_a, pos_a in teams:
        for team_b, pos_b in teams:
            if team_a == team_b:
                continue
            total = pos_a + pos_b
            if total <= 0:
                continue
            rows.append({
                'team_possession': pos_a,
                'opponent_possession': pos_b,
                'possession_gap': pos_a - pos_b,
                'target_possession': (pos_a / total) * 100.0,
            })
    return pd.DataFrame(rows)


def _plot_metric_axis(ax, labels, values, style):
    """Plot one metric trend line or a no-data placeholder on the given axis."""
    if any(not _is_nan(v) for v in values):
        ax.plot(
            labels, values, marker=style['marker'], color=style['color'],
            linewidth=2, markersize=8, label=style['label'],
        )
        _annotate_line(ax, labels, values)
        ax.set_title(style['title'])
        ax.set_ylabel(style['ylabel'])
        ax.set_ylim(0, 100)
    else:
        ax.text(
            0.5, 0.5, style['no_data_msg'], ha='center',
            va='center', transform=ax.transAxes,
        )
        ax.set_axis_off()
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)


class TeamAnalyzer:
    """Handles data acquisition, normalization, analytics, and reporting."""

    api_url_template = (
        'https://api.sportsdata.io/v4/soccer/scores/json/TeamSeasonStats/3/{season}'
    )
    competition_details_template = (
        'https://api.sportsdata.io/v4/soccer/scores/json/CompetitionDetails/{competition_id}'
    )
    default_seasons = (2025, 2026)
    default_team_name = 'Arsenal FC'

    def __init__(self):
        """Initialise the analyser with empty data structures."""
        self.team_data = None
        self.match_data = None

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt(value, spec='.2f'):
        """Format value using spec, returning 'N/A' for NaN/None."""
        if _is_nan(value):
            return 'N/A'
        try:
            return format(value, spec)
        except (TypeError, ValueError):
            return 'N/A'

    # ------------------------------------------------------------------
    # Fallback data
    # ------------------------------------------------------------------

    @staticmethod
    def _default_match_data(team_name='Arsenal FC'):
        """Return synthetic fallback data for offline use."""
        return pd.DataFrame({
            'date': pd.to_datetime(['2025-12-31', '2026-12-31']),
            'season_year': [2025, 2026],
            'team': [team_name] * 2,
            'possession': [56.8, 57.5],
            'shots': [495, 518],
            'shots_on_target': [172, 189],
        })

    # ------------------------------------------------------------------
    # HTTP / JSON helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_json(url, headers=None, params=None):
        """Fetch and return a parsed JSON payload from an HTTP endpoint."""
        if params:
            separator = '&' if '?' in url else '?'
            url = f"{url}{separator}{urlencode(params)}"
        req = Request(url, headers=headers or {})
        with urlopen(req, timeout=30) as response:
            payload = response.read().decode('utf-8', errors='ignore')
        return json.loads(payload)

    # ------------------------------------------------------------------
    # Payload normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_team_season_payload(payload):
        """Normalise TeamSeasonStats/Round payloads to a flat row list."""
        def _coerce(items, round_context=None):
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

        def _coerce_dict(data):
            for key in ('TeamSeasonStats', 'teamSeasonStats', 'data', 'Data'):
                value = data.get(key)
                if isinstance(value, list):
                    return _coerce(value)
                if isinstance(value, dict):
                    return _coerce([value])
            for key in ('Rounds', 'rounds'):
                rounds = data.get(key)
                if isinstance(rounds, list):
                    return _coerce(rounds)
            if isinstance(data.get('TeamSeasons'), list):
                context = {
                    'Season': data.get('Season'),
                    'SeasonType': data.get('SeasonType'),
                    'RoundId': data.get('RoundId'),
                    'RoundName': data.get('Name'),
                }
                return _coerce(data['TeamSeasons'], round_context=context)
            return _coerce([data])

        if isinstance(payload, list):
            return _coerce(payload)
        if isinstance(payload, dict):
            return _coerce_dict(payload)
        return []

    # ------------------------------------------------------------------
    # Environment / URL helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_env_file(env_path='.env'):
        """Load KEY=VALUE pairs from a .env file into the environment."""
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
        """Strip the 'key' query param from a URL; return (clean_url, key)."""
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
        """Return the first non-empty value for any of the candidate keys."""
        for key in keys:
            value = item.get(key)
            if value is not None and value != '':
                return value
        return np.nan

    @staticmethod
    def _team_matches(row_team_name, requested_team_name):
        """Case-insensitive exact match between two team name strings."""
        if pd.isna(row_team_name):
            return False
        return (
            str(row_team_name).strip().lower()
            == str(requested_team_name).strip().lower()
        )

    @staticmethod
    def _extract_team_season_row(team_stats, team_name='Arsenal FC'):
        """Convert a raw TeamSeasonStats dict into a normalised row dict."""
        source_team = TeamAnalyzer._pick_value(
            team_stats, 'Name', 'TeamName', 'Team', 'TeamKey'
        )
        if not TeamAnalyzer._team_matches(source_team, team_name):
            return None
        return _build_season_row(team_stats, TeamAnalyzer._pick_value, team_name)

    @classmethod
    def _url_for_season(cls, season_year):
        """Derive the TeamSeasonStats URL for a given season year."""
        return cls.api_url_template.format(season=season_year)

    # ------------------------------------------------------------------
    # Raw data logging
    # ------------------------------------------------------------------

    @staticmethod
    def _log_raw_api_data(season_year, raw_rows, team_name, log_path=None):
        """Write raw API rows for a season to a timestamped JSON log file."""
        if log_path is None:
            safe_name = team_name.lower().replace(' ', '_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path(__file__).parent / 'logs'
            log_dir.mkdir(exist_ok=True)
            log_path = log_dir / f'{safe_name}_api_data_{timestamp}.json'
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
            json.dumps({'api_log': existing}, indent=2, default=str),
            encoding='utf-8',
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_match_data_from_api(
        self, api_url, api_key, team_name=default_team_name, log_path=None
    ):
        """Fetch and parse TeamSeason rows for one season URL.

        Returns a DataFrame. Does not modify self.match_data.
        """
        cleaned_url, key_from_url = self._extract_api_key_from_url(api_url)
        resolved_api_key = api_key or key_from_url
        if not resolved_api_key:
            raise ValueError('A SportsData.io API key is required.')
        headers = _build_api_headers(resolved_api_key)
        payload = self._fetch_json(cleaned_url, headers=headers)
        normalized = self._normalize_team_season_payload(payload)
        raw_team_rows = [
            item for item in normalized
            if isinstance(item, dict)
            and self._team_matches(
                self._pick_value(item, 'Name', 'TeamName', 'Team', 'TeamKey'),
                team_name,
            )
        ]
        rows = [
            row for row in (
                self._extract_team_season_row(r, team_name=team_name)
                for r in normalized
            )
            if row
        ]
        if not rows:
            raise ValueError(f'No {team_name} data parsed from: {cleaned_url}')
        season_year = rows[0].get('season_year', 'unknown')
        self._log_raw_api_data(season_year, raw_team_rows, team_name=team_name, log_path=log_path)
        data_frame = pd.DataFrame(rows)
        _coerce_numeric_cols(data_frame)
        return data_frame

    def load_sample_data(
        self, api_key=None, seasons=default_seasons, team_name=default_team_name
    ):
        """Load data for each season and combine into self.match_data.

        Seasons that fail are skipped with a warning. Falls back to
        built-in sample data if all seasons fail or no key is configured.
        """
        self._load_env_file()
        TeamAnalyzer._log_buffer = {}
        resolved_api_key = api_key or os.getenv('SPORTSDATA_API_KEY')
        if not resolved_api_key:
            warnings.warn(
                'SPORTSDATA_API_KEY is not configured. Using fallback sample data.',
                RuntimeWarning,
            )
            self.match_data = self._default_match_data(team_name=team_name)
            return
        safe_name = team_name.lower().replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(__file__).parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        run_log_path = log_dir / f'{safe_name}_api_data_{timestamp}.json'

        season_frames = []
        for season_year in seasons:
            url = self._url_for_season(season_year)
            try:
                data_frame = self.load_match_data_from_api(
                    api_url=url, api_key=resolved_api_key,
                    team_name=team_name, log_path=run_log_path,
                )
                season_frames.append(data_frame)
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
    # Statistics
    # ------------------------------------------------------------------

    def calculate_basic_stats(self, team_name, season_year=None):
        """Return per-round average possession and shot accuracy for a team.

        If season_year is given, only rows for that season are used.
        """
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
            'avg_possession': (
                round(avg_possession, 2) if not pd.isna(avg_possession) else np.nan
            ),
            'shot_accuracy': (
                round(shot_accuracy, 2) if not pd.isna(shot_accuracy) else np.nan
            ),
        }, dtype='object')

    def predict_next_rounds_possession(self, team_name, opponents, season_year=None):
        """Predict possession split vs each opponent using a sklearn regressor.

        Returns a DataFrame with one row per opponent and columns:
        team, opponent, predicted_team_possession, predicted_opponent_possession.
        """
        if self.match_data is None or self.match_data.empty:
            raise ValueError('No match data loaded. Call load_sample_data() first.')
        if not opponents:
            raise ValueError('At least one opponent must be provided.')
        if 'possession' not in self.match_data.columns:
            raise ValueError("match_data is missing required column: 'possession'.")

        data_frame = self.match_data.copy()
        if season_year is not None and 'season_year' in data_frame.columns:
            data_frame = data_frame[data_frame['season_year'] == season_year]
        if data_frame.empty:
            raise ValueError(f'No data available for season {season_year}.')

        team_possession = (
            data_frame.groupby('team', as_index=True)['possession']
            .mean()
            .dropna()
            .to_dict()
        )
        if team_name not in team_possession:
            raise ValueError(f'{team_name} not found in available possession data.')
        missing_opponents = [
            opponent for opponent in opponents if opponent not in team_possession
        ]
        if missing_opponents:
            missing = ', '.join(str(item) for item in missing_opponents)
            raise ValueError(f'Opponent(s) not found in available possession data: {missing}')

        training_frame = _build_possession_training_frame(team_possession)
        if training_frame.empty:
            raise ValueError('Insufficient data to train possession prediction model.')

        model = None
        try:
            from sklearn.linear_model import LinearRegression
            features = training_frame[
                ['team_possession', 'opponent_possession', 'possession_gap']
            ]
            target = training_frame['target_possession']
            model = LinearRegression()
            model.fit(features, target)
        except ImportError:
            warnings.warn(
                'scikit-learn is not installed. Falling back to ratio-based '
                'possession prediction.',
                RuntimeWarning,
            )

        team_avg_possession = team_possession[team_name]
        predictions = []
        for opponent in opponents:
            opponent_avg_possession = team_possession[opponent]
            if model is not None:
                model_features = pd.DataFrame([{
                    'team_possession': team_avg_possession,
                    'opponent_possession': opponent_avg_possession,
                    'possession_gap': team_avg_possession - opponent_avg_possession,
                }])
                raw_prediction = float(model.predict(model_features)[0])
            else:
                total = team_avg_possession + opponent_avg_possession
                raw_prediction = 50.0 if total <= 0 else (team_avg_possession / total) * 100.0
            team_prediction = float(np.clip(raw_prediction, 0.0, 100.0))
            opponent_prediction = 100.0 - team_prediction
            predictions.append({
                'team': team_name,
                'opponent': opponent,
                'predicted_team_possession': round(team_prediction, 2),
                'predicted_opponent_possession': round(opponent_prediction, 2),
            })

        return pd.DataFrame(predictions)

    # ------------------------------------------------------------------
    # Competition / team discovery
    # ------------------------------------------------------------------

    def fetch_competition_details(self, api_key=None, competition_id=3):
        """Return a dict of {team_name: wikipedia_logo_url} for a competition."""
        self._load_env_file()
        resolved_api_key = api_key or os.getenv('SPORTSDATA_API_KEY')
        if not resolved_api_key:
            warnings.warn(
                'SPORTSDATA_API_KEY is not configured. Cannot fetch competition details.',
                RuntimeWarning,
            )
            return {}
        url = self.competition_details_template.format(competition_id=competition_id)
        headers = {
            'Accept': 'application/json',
            'Ocp-Apim-Subscription-Key': resolved_api_key,
        }
        try:
            payload = self._fetch_json(url, headers=headers)
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            warnings.warn(
                f'Failed to fetch competition details ({exc}). No logos will be shown.',
                RuntimeWarning,
            )
            return {}
        logos = {}
        teams = payload.get('Teams') or payload.get('teams') or []
        for team in teams:
            if not isinstance(team, dict):
                continue
            name = team.get('Name') or team.get('TeamName') or team.get('ShortName')
            logo_url = team.get('WikipediaLogoUrl') or team.get('wikipediaLogoUrl')
            if name:
                logos[str(name).strip()] = logo_url or None
        return logos

    def fetch_all_teams(self, api_key=None, seasons=default_seasons):
        """Return a sorted, deduplicated list of team names from the API.

        Queries all seasons in `seasons`. Does not modify match_data.
        """
        self._load_env_file()
        resolved_api_key = api_key or os.getenv('SPORTSDATA_API_KEY')
        if not resolved_api_key:
            warnings.warn(
                'SPORTSDATA_API_KEY is not configured. Returning fallback team list.',
                RuntimeWarning,
            )
            return [self.default_team_name]
        team_names = set()
        for season_year in seasons:
            url = self._url_for_season(season_year)
            cleaned_url, key_from_url = self._extract_api_key_from_url(url)
            resolved_key = resolved_api_key or key_from_url
            headers = _build_api_headers(resolved_key)
            try:
                payload = self._fetch_json(cleaned_url, headers=headers)
                rows = self._normalize_team_season_payload(payload)
                for item in rows:
                    if not isinstance(item, dict):
                        continue
                    name = self._pick_value(item, 'Name', 'TeamName', 'Team', 'TeamKey')
                    if not _is_nan(name):
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
            return [self.default_team_name]
        return sorted(team_names)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_report(self, team_name, seasons=default_seasons):
        """Return a side-by-side text report of metrics across seasons."""
        season_stats = {}
        available_seasons = []
        for yr in seasons:
            if (
                'season_year' in self.match_data.columns
                and yr not in self.match_data['season_year'].values
            ):
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
            for year in available_seasons:
                raw = season_stats[year].get(key, np.nan)
                cell = self._fmt(raw, spec) + suffix if not _is_nan(raw) else 'N/A'
                values += cell.rjust(col_w)
            return f'{label:<{label_w}}{values}'

        lines = [
            f'\nPerformance Report — {team_name}',
            f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            '(Values are averages per round across each season)',
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
    # Chart
    # ------------------------------------------------------------------

    def plot_performance_trends(self, team_name, seasons=default_seasons, logo_url=None):
        """Return a matplotlib figure with possession and shot accuracy trends.

        If logo_url is provided, the team logo is shown in the title area.
        """
        available_seasons, possession_vals, shot_accuracy_vals = (
            _collect_season_stats(self, team_name, seasons)
        )
        season_labels = [str(yr) for yr in available_seasons]
        logo_img = _fetch_logo_pil(logo_url, size=(52, 52)) if logo_url else None
        has_logo = logo_img is not None
        title_top = 0.86

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        _plot_metric_axis(ax1, season_labels, possession_vals, {
            'marker': 'o',
            'color': 'green',
            'label': 'Avg Possession %',
            'title': 'Average Possession % per Round',
            'ylabel': 'Possession (%)',
            'no_data_msg': 'Possession data not available',
        })
        _plot_metric_axis(ax2, season_labels, shot_accuracy_vals, {
            'marker': 's',
            'color': 'darkorange',
            'label': 'Avg Shot Accuracy %',
            'title': 'Average Shot Accuracy % per Round',
            'ylabel': 'Shot Accuracy (%)',
            'no_data_msg': 'Shot accuracy data not available',
        })

        plt.tight_layout(rect=[0, 0, 1, title_top])

        title_x = _add_chart_logo(fig, logo_img, title_top) if has_logo else 0.5

        fig.suptitle(
            f'{team_name} — Season-by-Season Performance Metrics',
            fontsize=14, fontweight='bold',
            y=(title_top + 1.0) / 2,
            x=title_x,
        )
        return fig
