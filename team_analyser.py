import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import URLError
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

    @staticmethod
    def _fetch_html(url):
        """Fetch HTML content from a URL with browser-like headers."""
        req = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urlopen(req, timeout=30) as response:
            return response.read().decode("utf-8", errors="ignore")

    @staticmethod
    def _extract_match_ids(html):
        """Extract unique Premier League match IDs from link paths."""
        match_ids = re.findall(r'/en/match/(\d+)(?:/|"|\?)', html)
        ordered_unique = []
        seen = set()
        for match_id in match_ids:
            if match_id in seen:
                continue
            seen.add(match_id)
            ordered_unique.append(match_id)
        return ordered_unique

    @staticmethod
    def _extract_stat_pair(html, label, include_label=None, exclude_label=None):
        """Extract a pair of numeric values for home/away teams from a stat row."""
        pattern = re.compile(
            rf'{label}(.{{0,300}}?)(\d+(?:\.\d+)?)\s*%?(.{{0,120}}?)(\d+(?:\.\d+)?)\s*%?',
            flags=re.IGNORECASE | re.DOTALL,
        )
        for match in pattern.finditer(html):
            snippet = match.group(0)
            snippet_lower = snippet.lower()
            if include_label and include_label.lower() not in snippet_lower:
                continue
            if exclude_label and exclude_label.lower() in snippet_lower:
                continue
            return float(match.group(2)), float(match.group(4))
        return np.nan, np.nan

    def _fetch_match_stats(self, match_id, venue):
        """Fetch possession/shooting stats from the match stats page."""
        stats_url = f"https://www.premierleague.com/en/match/{match_id}/stats"
        try:
            html = self._fetch_html(stats_url)
        except (URLError, TimeoutError, ValueError):
            return {}

        possession_home, possession_away = self._extract_stat_pair(html, 'possession')
        shots_home, shots_away = self._extract_stat_pair(html, 'total shots|shots', exclude_label='on target')
        shots_on_target_home, shots_on_target_away = self._extract_stat_pair(html, 'shots on target')

        is_home = str(venue).lower() == 'home'
        return {
            'possession': possession_home if is_home else possession_away,
            'shots': shots_home if is_home else shots_away,
            'shots_on_target': shots_on_target_home if is_home else shots_on_target_away,
        }

    def _enrich_with_match_stats(self, match_data, matches_html):
        """Populate possession and shooting columns from per-match stats pages."""
        if match_data.empty:
            return match_data

        if 'match_id' not in match_data.columns or match_data['match_id'].isna().all():
            ids = self._extract_match_ids(matches_html)
            if ids:
                count = min(len(ids), len(match_data))
                match_data = match_data.copy()
                match_data['match_id'] = pd.Series(ids[:count], index=match_data.index[:count])

        if 'possession' not in match_data.columns:
            match_data['possession'] = np.nan
        if 'shots' not in match_data.columns:
            match_data['shots'] = np.nan
        if 'shots_on_target' not in match_data.columns:
            match_data['shots_on_target'] = np.nan

        stats_cache = {}
        for idx, row in match_data.iterrows():
            match_id = row.get('match_id')
            if pd.isna(match_id):
                continue
            match_id = str(int(match_id)) if isinstance(match_id, (int, float)) and not pd.isna(match_id) else str(match_id)

            if match_id not in stats_cache:
                stats_cache[match_id] = self._fetch_match_stats(match_id, row.get('venue', 'Home'))

            stats = stats_cache[match_id]
            for key in ('possession', 'shots', 'shots_on_target'):
                if key in stats and pd.notna(stats[key]):
                    match_data.at[idx, key] = stats[key]

        return match_data

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
                    "match_id": np.nan,
                })

        return pd.DataFrame(rows)

    def _parse_match_cards_from_html(self, html):
        """Parse fixtures from rendered match-card HTML fragments."""
        card_pattern = re.compile(r'<li[^>]*class="[^"]*match-card[^"]*"[^>]*>(.*?)</li>', re.DOTALL | re.IGNORECASE)
        cards = card_pattern.findall(html)

        rows = []
        for card in cards:
            teams = re.findall(
                r'<span[^>]*data-testid="matchCardTeamFullName"[^>]*>(.*?)</span>',
                card,
                flags=re.DOTALL | re.IGNORECASE,
            )
            teams = [re.sub(r'<[^>]+>', '', team).strip() for team in teams if team.strip()]
            if len(teams) < 2:
                continue

            score_match = re.search(
                r'<span[^>]*data-testid="matchCardScore"[^>]*>(.*?)</span>',
                card,
                flags=re.DOTALL | re.IGNORECASE,
            )
            score_text = re.sub(r'<[^>]+>', '', score_match.group(1)).strip() if score_match else ''
            home_score, away_score = self._extract_score(score_text)

            date_match = re.search(r'<time[^>]*datetime="([^"]+)"', card, flags=re.IGNORECASE)
            if not date_match:
                date_match = re.search(r'data-testid="matchCardDate"[^>]*>(.*?)<', card, flags=re.DOTALL | re.IGNORECASE)
            date_raw = date_match.group(1).strip() if date_match else None

            match_link = re.search(r'href="([^"]*/en/match/(\d+)[^"]*)"', card, flags=re.IGNORECASE)
            match_id = match_link.group(2) if match_link else np.nan

            home, away = teams[0], teams[1]
            names = {home.lower(), away.lower()}
            if 'arsenal' not in names:
                continue

            is_arsenal_home = home.lower() == 'arsenal'
            rows.append({
                'date': pd.to_datetime(date_raw, errors='coerce'),
                'team': 'Arsenal',
                'opponent': away if is_arsenal_home else home,
                'venue': 'Home' if is_arsenal_home else 'Away',
                'goals_scored': home_score if is_arsenal_home else away_score,
                'goals_conceded': away_score if is_arsenal_home else home_score,
                'match_id': match_id,
            })

        if not rows:
            team_iter = list(re.finditer(r'<span[^>]*data-testid="matchCardTeamFullName"[^>]*>(.*?)</span>', html, flags=re.DOTALL | re.IGNORECASE))
            score_iter = list(re.finditer(r'<span[^>]*data-testid="matchCardScore"[^>]*>(.*?)</span>', html, flags=re.DOTALL | re.IGNORECASE))

            teams = [
                (m.start(), re.sub(r'<[^>]+>', '', m.group(1)).strip())
                for m in team_iter
                if re.sub(r'<[^>]+>', '', m.group(1)).strip()
            ]

            for score_m in score_iter:
                score_idx = score_m.start()
                before = [team for pos, team in teams if pos < score_idx]
                after = [team for pos, team in teams if pos > score_idx]
                if not before or not after:
                    continue

                home, away = before[-1], after[0]
                names = {home.lower(), away.lower()}
                if 'arsenal' not in names:
                    continue

                score_text = re.sub(r'<[^>]+>', '', score_m.group(1)).strip()
                home_score, away_score = self._extract_score(score_text)
                is_arsenal_home = home.lower() == 'arsenal'
                rows.append({
                    'date': pd.NaT,
                    'team': 'Arsenal',
                    'opponent': away if is_arsenal_home else home,
                    'venue': 'Home' if is_arsenal_home else 'Away',
                    'goals_scored': home_score if is_arsenal_home else away_score,
                    'goals_conceded': away_score if is_arsenal_home else home_score,
                    'match_id': np.nan,
                })

        parsed = pd.DataFrame(rows)
        if parsed.empty:
            return parsed

        return parsed.dropna(subset=['opponent']).drop_duplicates(subset=['date', 'opponent', 'goals_scored', 'goals_conceded'])

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
                    'match_id': item.get('id') or item.get('matchId') or item.get('gameId') or np.nan,
                })

        parsed = pd.DataFrame(rows)
        if parsed.empty:
            return parsed

        parsed = parsed.dropna(subset=['opponent']).drop_duplicates(subset=['date', 'opponent', 'goals_scored', 'goals_conceded'])
        return parsed

    def load_match_data_from_web(self, url="https://www.premierleague.com/en/clubs/3/arsenal/matches"):
        """Load Arsenal matches from the official Premier League page and replace `match_data`."""
        try:
            html = self._fetch_html(url)
        except (URLError, TimeoutError, ValueError):
            html = ''

        match_data = self._parse_ld_json_matches(html)

        if match_data.empty:
            match_data = self._parse_embedded_match_json(html)

        if match_data.empty:
            match_data = self._parse_match_cards_from_html(html)

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
                        "match_id": np.nan,
                    })

        if match_data.empty:
            warnings.warn(
                "Unable to parse match data from the Premier League page. Falling back to default sample data.",
                RuntimeWarning,
            )
            match_data = self._default_match_data()

        match_data = self._enrich_with_match_stats(match_data, html)
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
