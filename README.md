# Football Team Analyser

A lightweight Python project for analysing football (soccer) team performance using **pandas** and visualising trends with **matplotlib**.

## Features

- Fetches season-aggregate data from the SportsData.io `TeamSeasonStats` endpoint across multiple seasons (default: 2025 and 2026).
- At startup, extracts all team names available across both seasons from the API and presents them in a **dropdown selection window** — no hardcoded team name required.
- Calculates per-round average performance metrics:
  - Average possession (sum of all round values ÷ number of rounds)
  - Shot accuracy (per-round `shots_on_target / shots × 100`, averaged across rounds)
- Generates a side-by-side text report comparing metrics across seasons.
- Plots performance trends over seasons (possession and shot accuracy) on a single chart.
- Writes a full JSON log of all raw API data for the selected team, overwritten on each run.

## ⚠️ SportsData.io Licence Limitations

This project relies on the [SportsData.io](https://sportsdata.io) API. Access to data is subject to your subscription plan, and the following limitations apply:

**Seasons:** Only seasons covered by your licence are accessible. The default configuration targets **2025 and 2026**. Requesting a season not included in your plan will result in an empty response or an error, and that season will be skipped with a warning.

**Competitions:** The API URL uses **competition ID 3** (UEFA Champions League). If your licence does not include this competition, the requests will fail. You can change the competition ID in `API_URL_TEMPLATE` inside `team_analyser.py` to match a competition covered by your plan. Refer to the [SportsData.io soccer documentation](https://sportsdata.io/developers/api-documentation/soccer) for valid competition IDs.

**Teams:** The team dropdown is populated dynamically from whatever the API returns for your licensed seasons and competition. If a team does not appear, it is either not present in that competition or not covered by your licence.

**Free / trial plans** typically provide access to a limited set of competitions and historical seasons only. If all requests fail, the application falls back to built-in sample data so it can still run offline.

## Project Structure

```text
.
├── team_analyser.py        # Main script containing TeamAnalyzer and GUI picker
├── .env                    # API credentials (not committed — see below)
├── Python.gitignore
└── JetBrains.gitIgnore
```

On each run, a JSON log file is created in the same directory as the script:

```text
<team_name_lowercase>_api_data.json   # e.g. arsenal_fc_api_data.json
```

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- tkinter *(included in the Python standard library — no separate install needed)*

Install third-party dependencies:

```bash
pip install pandas numpy matplotlib
```

## Environment Configuration

Create a `.env` file in the project root (this file is gitignored and must never be committed):

```dotenv
SPORTSDATA_API_KEY=your_api_key_here
```

The `SPORTSDATA_MATCHES_URL` variable is no longer used. Season URLs are now derived automatically from `API_URL_TEMPLATE` in the script.

You can also export the key as a shell variable:

```bash
export SPORTSDATA_API_KEY="your_api_key_here"
```

> **Security note:** Never commit your real API key to the repository, including in the README. If a key has been accidentally exposed, regenerate it immediately on the SportsData.io dashboard.

## Quick Start

```bash
python team_analyser.py
```

What happens on launch:

1. The API is queried for all teams available in seasons 2025 and 2026 under the configured competition.
2. A dropdown window opens listing all unique team names found (deduplicated across both seasons).
3. Select a team and click **Analyse**.
4. The performance report is printed to the console.
5. A trend chart opens showing average possession and shot accuracy per round across both seasons.
6. A JSON log file is written to the project directory with all raw API data for the selected team.

## Usage Example

```python
from team_analyser import TeamAnalyzer
import matplotlib.pyplot as plt

analyzer = TeamAnalyzer()

# Fetch available teams (optional — used by the GUI)
teams = analyzer.fetch_all_teams(seasons=(2025, 2026))
print(teams)

# Load data for a specific team
analyzer.load_sample_data(seasons=(2025, 2026), team_name='Arsenal FC')

# Print side-by-side report
print(analyzer.generate_report('Arsenal FC', seasons=(2025, 2026)))

# Show trend chart
fig = analyzer.plot_performance_trends('Arsenal FC', seasons=(2025, 2026))
plt.show()
```

## Core API

### `TeamAnalyzer.fetch_all_teams(api_key, seasons)`
Queries the API for all seasons in `seasons` and returns a sorted, deduplicated list of team names found across all of them. Used internally to populate the dropdown. Does not modify `match_data`.

### `TeamAnalyzer.load_sample_data(api_key, seasons, team_name)`
Fetches data for the given team across all seasons in `seasons`. Each season URL is derived automatically from `API_URL_TEMPLATE`. Seasons that fail to load are skipped with a warning. If all seasons fail, falls back to built-in sample data. Also resets the log buffer so each run produces a fresh log file.

### `TeamAnalyzer.load_match_data_from_api(api_url, api_key, team_name)`
Fetches and parses `TeamSeason` rows for a single season URL. Returns a DataFrame without modifying `self.match_data`. Also writes raw API rows for the team to the JSON log file.

### `TeamAnalyzer.calculate_basic_stats(team_name, season_year)`
Returns a `pandas.Series` with per-round average performance metrics for the given team. If `season_year` is provided, only rows for that season are used. Metrics returned: `avg_possession`, `shot_accuracy`.

### `TeamAnalyzer.generate_report(team_name, seasons)`
Prints a side-by-side columnar text report comparing `avg_possession` and `shot_accuracy` across all seasons in `seasons`. Values are averages per round.

### `TeamAnalyzer.plot_performance_trends(team_name, seasons)`
Returns a matplotlib figure with two line charts — average possession % and average shot accuracy % — one data point per season, with annotated values on each point.

## Changing Seasons or Competition

To analyse different seasons, update `DEFAULT_SEASONS` in the class and the `SEASONS` tuple in the `__main__` block:

```python
DEFAULT_SEASONS = (2024, 2025)   # in the class
SEASONS = (2024, 2025)           # in __main__
```

To change the competition, update the competition ID in `API_URL_TEMPLATE`:

```python
API_URL_TEMPLATE = 'https://api.sportsdata.io/v4/soccer/scores/json/TeamSeasonStats/{competition_id}/{season}'
```

Refer to the SportsData.io documentation for the list of competition IDs available under your plan.

## Notes

- All metrics are **per-round averages**, not season totals. This reflects the structure of the `TeamSeasonStats` endpoint, which returns one row per team per round rather than a single season aggregate.
- The `pick_team_gui()` function requires a display (i.e. it will not run in a headless environment). In such cases, call `load_sample_data()` directly with a team name.
- The JSON log file is **overwritten on every run**. If you need to retain logs across runs, copy or rename the file before re-running.

## License

No licence file is currently included in this repository.