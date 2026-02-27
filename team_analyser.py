import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from urllib.request import Request, urlopen
from io import StringIO
import json
import re
import warnings


class TeamAnalyzer:
    def __init__(self):
        """Initialize the analyzer with empty data structures"""
        self.team_data = None
        self.match_data = None

    def load_sample_data(self):
        """Load sample match data"""
        self.load_match_data_from_web()

    @staticmethod
    def _default_match_data():
        """Static fallback data used only when web parsing is unavailable."""
        return pd.DataFrame({
            'date': pd.to_datetime(['2024-01-15', '2024-01-22', '2024-01-29', '2024-02-05', '2024-02-12']),
            'team': ['Arsenal'] * 5,
            'opponent': ['Liverpool', 'Chelsea', 'Manchester City', 'Tottenham', 'Newcastle'],
            'goals_scored': [2, 1, 3, 2, 4],
            'goals_conceded': [1, 2, 0, 1, 1],
            'possession': [55, 48, 62, 51, 58],
            'shots': [14, 11, 17, 13, 18],
            'shots_on_target': [6, 4, 8, 5, 9]
        })

    @staticmethod
    def _extract_score(value):
        """Extract scoreline from strings like `2-1`, `2 : 1` or `Arsenal 2-1 Chelsea`."""
        if not isinstance(value, str):
            return np.nan, np.nan

        score_match = re.search(r"(\d+)\s*[-:]\s*(\d+)", value)
        if not score_match:
            return np.nan, np.nan

        return int(score_match.group(1)), int(score_match.group(2))

    def _parse_ld_json_matches(self, html):
        """Parse SportsEvent entries from JSON-LD blocks in the page."""
        scripts = re.findall(
            r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>',
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )

        rows = []
        for raw_json in scripts:
            try:
                payload = json.loads(raw_json.strip())
            except json.JSONDecodeError:
                continue

            entries = payload if isinstance(payload, list) else [payload]
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if entry.get("@type") != "SportsEvent":
                    continue

                home = entry.get("homeTeam", {}).get("name")
                away = entry.get("awayTeam", {}).get("name")
                start_date = entry.get("startDate")
                score_text = entry.get("name", "")
                home_score, away_score = self._extract_score(score_text)

                if not home or not away:
                    continue

                is_arsenal_home = home.lower() == "arsenal"
                rows.append({
                    "date": pd.to_datetime(start_date, errors="coerce"),
                    "team": "Arsenal",
                    "opponent": away if is_arsenal_home else home,
                    "venue": "Home" if is_arsenal_home else "Away",
                    "goals_scored": home_score if is_arsenal_home else away_score,
                    "goals_conceded": away_score if is_arsenal_home else home_score,
                })

        return pd.DataFrame(rows)

    def _parse_embedded_match_json(self, html):
        """Parse fixtures from non-JSON-LD script blobs used by modern web apps."""
        scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, flags=re.DOTALL | re.IGNORECASE)

        candidates = []
        for script in scripts:
            script = script.strip()
            if not script:
                continue

            if script.startswith('{') or script.startswith('['):
                candidates.append(script)

            for match in re.finditer(r'([\[{].*[\]}])', script, flags=re.DOTALL):
                blob = match.group(1).strip().rstrip(';')
                if blob.startswith(('{', '[')):
                    candidates.append(blob)

        def walk(node):
            if isinstance(node, dict):
                yield node
                for value in node.values():
                    yield from walk(value)
            elif isinstance(node, list):
                for item in node:
                    yield from walk(item)

        rows = []
        for payload_text in candidates:
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                continue

            for item in walk(payload):
                home = away = None
                if isinstance(item.get('homeTeam'), dict) and isinstance(item.get('awayTeam'), dict):
                    home = item.get('homeTeam', {}).get('name')
                    away = item.get('awayTeam', {}).get('name')
                elif isinstance(item.get('teams'), dict):
                    home = item.get('teams', {}).get('home', {}).get('name')
                    away = item.get('teams', {}).get('away', {}).get('name')

                if not home or not away:
                    continue

                names = {home.lower(), away.lower()}
                if 'arsenal' not in names:
                    continue

                date_raw = (
                    item.get('startDate')
                    or item.get('kickoff')
                    or item.get('date')
                    or item.get('matchDate')
                )

                home_score = item.get('homeScore')
                away_score = item.get('awayScore')

                if home_score is None or away_score is None:
                    score_text = item.get('score') or item.get('result') or item.get('name', '')
                    home_score, away_score = self._extract_score(str(score_text))

                is_arsenal_home = home.lower() == 'arsenal'
                rows.append({
                    'date': pd.to_datetime(date_raw, errors='coerce'),
                    'team': 'Arsenal',
                    'opponent': away if is_arsenal_home else home,
                    'venue': 'Home' if is_arsenal_home else 'Away',
                    'goals_scored': home_score if is_arsenal_home else away_score,
                    'goals_conceded': away_score if is_arsenal_home else home_score,
                })

        parsed = pd.DataFrame(rows)
        if parsed.empty:
            return parsed

        parsed = parsed.dropna(subset=['opponent']).drop_duplicates(subset=['date', 'opponent', 'goals_scored', 'goals_conceded'])
        return parsed

    def load_match_data_from_web(self, url="https://www.premierleague.com/en/clubs/3/arsenal/matches"):
        """Load Arsenal matches from the official Premier League page and replace `match_data`."""
        req = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urlopen(req, timeout=30) as response:
            html = response.read().decode("utf-8", errors="ignore")

        match_data = self._parse_ld_json_matches(html)

        if match_data.empty:
            match_data = self._parse_embedded_match_json(html)

        if match_data.empty:
            # Optional fallback when JSON-LD is unavailable.
            # Some environments do not have the optional html parser deps (e.g. lxml).
            try:
                tables = pd.read_html(StringIO(html))
            except (ImportError, ValueError):
                tables = []

            if tables:
                table = tables[0].copy()
                table.columns = [str(col).strip().lower().replace(" ", "_") for col in table.columns]

                date_col = next((col for col in table.columns if "date" in col), None)
                opp_col = next((col for col in table.columns if "opponent" in col or "against" in col), None)
                score_col = next((col for col in table.columns if "score" in col or "result" in col), None)

                if date_col and opp_col:
                    goals = table[score_col].apply(self._extract_score) if score_col else []
                    goals_scored = [x[0] for x in goals] if score_col else np.nan
                    goals_conceded = [x[1] for x in goals] if score_col else np.nan
                    match_data = pd.DataFrame({
                        "date": pd.to_datetime(table[date_col], errors="coerce"),
                        "team": "Arsenal",
                        "opponent": table[opp_col],
                        "goals_scored": goals_scored,
                        "goals_conceded": goals_conceded,
                    })

        if match_data.empty:
            warnings.warn(
                "Unable to parse match data from the Premier League page. Falling back to default sample data.",
                RuntimeWarning,
            )
            match_data = self._default_match_data()

        self.match_data = match_data.sort_values("date").reset_index(drop=True)

    def calculate_basic_stats(self, team_name):
        """Calculate basic team statistics"""
        team_matches = self.match_data[self.match_data['team'] == team_name]

        stats = {
            'matches_played': len(team_matches),
            'goals_scored': team_matches.get('goals_scored', pd.Series(dtype=float)).sum(min_count=1),
            'goals_conceded': team_matches.get('goals_conceded', pd.Series(dtype=float)).sum(min_count=1),
            'goal_difference': team_matches.get('goals_scored', pd.Series(dtype=float)).sum(min_count=1) - team_matches.get('goals_conceded', pd.Series(dtype=float)).sum(min_count=1),
            'avg_possession': team_matches['possession'].mean() if 'possession' in team_matches else np.nan,
            'shot_accuracy': (team_matches['shots_on_target'].sum() / team_matches['shots'].sum() * 100) if {'shots_on_target', 'shots'}.issubset(team_matches.columns) else np.nan,
            'goals_per_game': team_matches.get('goals_scored', pd.Series(dtype=float)).mean(),
            'clean_sheets': len(team_matches[team_matches.get('goals_conceded', pd.Series(dtype=float)) == 0]) if 'goals_conceded' in team_matches else np.nan
        }

        return pd.Series(stats)

    def plot_performance_trends(self, team_name):
        """Plot performance trends over time"""
        team_matches = self.match_data[self.match_data['team'] == team_name]

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot goals
        ax1.plot(team_matches['date'], team_matches.get('goals_scored', pd.Series(index=team_matches.index, dtype=float)),
                 label='Goals Scored', marker='o')
        ax1.plot(team_matches['date'], team_matches.get('goals_conceded', pd.Series(index=team_matches.index, dtype=float)),
                 label='Goals Conceded', marker='o')
        ax1.set_title(f'{team_name} Goal Performance Over Time')
        ax1.legend()
        ax1.grid(True)

        # Plot possession
        if 'possession' in team_matches:
            ax2.plot(team_matches['date'], team_matches['possession'],
                     label='Possession %', marker='o', color='green')
            ax2.set_title(f'{team_name} Possession Over Time')
            ax2.legend()
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, 'Possession data is not available for web-scraped matches',
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_axis_off()

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
