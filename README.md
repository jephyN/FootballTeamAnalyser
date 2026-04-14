# Football Team Analyser

A lightweight Python project for analysing football (soccer) team performance using **pandas** and visualising trends with **matplotlib**.

## Features

- Fetches season-aggregate data from the SportsData.io `TeamSeasonStats` endpoint across multiple seasons (default: 2025 and 2026).
- At startup, fetches all teams active in the **2026 season** from the API and presents them in a **dropdown selection window** — no hardcoded team name required.
- Displays team logos in the dropdown (loaded in the background) and alongside the chart title, sourced from Wikimedia.
- Calculates per-round average performance metrics:
  - Average possession
  - Shot accuracy (per-round `shots_on_target / shots × 100`, averaged across rounds)
- Generates a side-by-side text report comparing metrics across seasons.
- Plots performance trends over seasons (possession and shot accuracy) on a single chart.
- Writes a timestamped JSON log of all raw API data for the selected team on each run.

## ⚠️ SportsData.io Licence Limitations

This project relies on the [SportsData.io](https://sportsdata.io) API. Access to data is subject to your subscription plan, and the following limitations apply:

**Seasons:** Only seasons covered by your licence are accessible. The default configuration targets **2025 and 2026**. Requesting a season not included in your plan will result in an empty response or an error, and that season will be skipped with a warning.

**Competitions:** The API URL uses **competition ID 3** (UEFA Champions League). If your licence does not include this competition, the requests will fail. You can change the competition ID in `API_URL_TEMPLATE` inside `team_analyser.py` to match a competition covered by your plan. Refer to the [SportsData.io soccer documentation](https://sportsdata.io/developers/api-documentation/soccer) for valid competition IDs.

**Teams:** The dropdown is populated from whatever the API returns for the 2026 season. If a team does not appear, it is either not present in that competition or not covered by your licence.

**Free / trial plans** typically provide access to a limited set of competitions and historical seasons only. If all requests fail, the application falls back to built-in sample data so it can still run offline.

## Project Structure

```text
.
├── main.py             # Entry point — orchestrates the startup sequence
├── team_analyser.py    # TeamAnalyzer class — data loading, stats, report, chart
├── gui.py              # Team picker GUI (tkinter, background icon loading)
├── logo_utils.py       # Wikimedia logo URL resolution and PIL image fetching
├── tests/
│   ├── conftest.py          # sys.path setup for pytest
│   ├── test_team_analyser.py
│   └── test_logo_utils.py
├── logs/               # Timestamped JSON log files (created automatically)
├── .env                # API credentials (not committed — see below)
├── Python.gitignore
└── JetBrains.gitIgnore
```

On each run, a timestamped JSON log file is created inside the `logs/` folder (created automatically if it doesn't exist). All seasons fetched in the same run are combined into a single file, organised by year:

```text
logs/<team_name_lowercase>_api_data_<YYYYMMDD_HHMMSS>.json
# e.g. logs/arsenal_fc_api_data_20260228_134500.json
```

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- scikit-learn *(for possession prediction vs upcoming opponents)*
- Pillow *(for team logos in the dropdown and chart)*
- tkinter *(included in the Python standard library — no separate install needed)*

Install third-party dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn Pillow
```

## Environment Configuration

Create a `.env` file in the project root (this file is gitignored and must never be committed):

```dotenv
SPORTSDATA_API_KEY=your_api_key_here
```

You can also export the key as a shell variable:

```bash
export SPORTSDATA_API_KEY="your_api_key_here"
```

> **Security note:** Never commit your real API key to the repository. If a key has been accidentally exposed, regenerate it immediately on the SportsData.io dashboard.

## Quick Start

```bash
python main.py
```

What happens on launch:

1. The API is queried for all teams available in the **2026 season** under the configured competition.
2. Logo URLs are fetched for all teams in the competition.
3. A dropdown window opens listing all unique team names. Team logos load in the background as small icons beside each name.
4. Select a team and click **Analyse** (or double-click / press Enter).
5. The performance report is printed to the console.
6. A trend chart opens showing average possession and shot accuracy per round across **both** seasons (2025 and 2026), with the team logo displayed in the title area.
7. A timestamped JSON log file is written to the `logs/` folder with all raw API data for the selected team, organised by season year.

## Usage Example

```python
from team_analyser import TeamAnalyzer
import matplotlib.pyplot as plt

analyzer = TeamAnalyzer()

# Fetch teams active in 2026 (used by the GUI dropdown)
teams = analyzer.fetch_all_teams(seasons=(2026,))
print(teams)

# Load data for a specific team across both seasons
analyzer.load_sample_data(seasons=(2025, 2026), team_name='Arsenal FC')

# Print side-by-side report
print(analyzer.generate_report('Arsenal FC', seasons=(2025, 2026)))

# Show trend chart (with optional logo)
logos = analyzer.fetch_competition_details()
logo_url = logos.get('Arsenal FC')
fig = analyzer.plot_performance_trends('Arsenal FC', seasons=(2025, 2026), logo_url=logo_url)
plt.show()
```

## Core API

### `TeamAnalyzer.fetch_all_teams(api_key, seasons)`
Queries the API for all seasons in `seasons` and returns a sorted, deduplicated list of team names. Used internally to populate the dropdown. Does not modify `match_data`.

### `TeamAnalyzer.fetch_competition_details(api_key, competition_id)`
Fetches the competition metadata endpoint and returns a `dict` of `{team_name: wikipedia_logo_url}`. Used to source logo URLs for both the dropdown icons and the chart title. Teams without a logo URL are included with a `None` value.

### `TeamAnalyzer.load_sample_data(api_key, seasons, team_name)`
Fetches data for the given team across all seasons in `seasons`. Seasons that fail to load are skipped with a warning. If all seasons fail, falls back to built-in sample data.

### `TeamAnalyzer.load_match_data_from_api(api_url, api_key, team_name, log_path)`
Fetches and parses `TeamSeason` rows for a single season URL. Returns a DataFrame without modifying `self.match_data`. Also writes raw API rows to the timestamped JSON log file.

### `TeamAnalyzer.calculate_basic_stats(team_name, season_year)`
Returns a `pandas.Series` with `avg_possession` and `shot_accuracy` for the given team, optionally filtered to a specific season.

### `TeamAnalyzer.generate_report(team_name, seasons)`
Returns a side-by-side columnar text report comparing `avg_possession` and `shot_accuracy` across all seasons in `seasons`.

### `TeamAnalyzer.predict_next_rounds_possession(team_name, opponents, season_year)`
Predicts possession split for upcoming opponents using a scikit-learn linear-regression model trained on historical team possession values. Returns a DataFrame with both values constrained to a full split: `team % + opponent % = 100 %`. If scikit-learn is not installed, the method falls back to a ratio-based estimator and emits a runtime warning.

### `TeamAnalyzer.plot_performance_trends(team_name, seasons, logo_url)`
Returns a matplotlib figure with two line charts — average possession % and average shot accuracy % — one data point per season. If `logo_url` is provided, the team logo is displayed in the top-left of the chart title area.

### `pick_team_gui(team_list, logos)` *(in `gui.py`)*
Opens a tkinter window with a scrollable list of team names. Team logos (sourced from the `logos` dict) are loaded in background threads and displayed as 24×24 icons beside each name. Returns the selected team name, or `None` if the window is closed without confirming.

### `_fetch_logo_pil(url, size)` *(in `logo_utils.py`)*
Downloads a logo from a Wikimedia URL and returns a PIL `Image` scaled to `size`. Uses the `thumb.php` REST endpoint to avoid CDN step-size restrictions. Retries up to 3 times on HTTP 429.

## Changing Seasons or Competition

To analyse different seasons, update `DEFAULT_SEASONS` in `team_analyser.py` and `SEASONS` in `main.py`:

```python
# team_analyser.py
DEFAULT_SEASONS = (2024, 2025)

# main.py
SEASONS = (2024, 2025)
```

To change the competition, update the competition ID in `API_URL_TEMPLATE` in `team_analyser.py`:

```python
API_URL_TEMPLATE = 'https://api.sportsdata.io/v4/soccer/scores/json/TeamSeasonStats/{competition_id}/{season}'
```

Refer to the SportsData.io documentation for the list of competition IDs available under your plan.

## Notes

- All metrics are **per-round averages**, not season totals.
- The dropdown and chart logos require **Pillow** (`pip install Pillow`). Without it, logos are silently skipped and grey placeholders are shown in the dropdown.
- `pick_team_gui()` requires a display and will not run in a headless environment. In such cases, call `load_sample_data()` directly with a team name.
- Each run produces a **new timestamped log file** inside the `logs/` folder rather than overwriting the previous one. Multiple seasons fetched in the same run are stored together in that file, keyed by year. The folder is created automatically on first run.

## License

No licence file is currently included in this repository.
