# FootballTeamAnalyser

A lightweight Python project for analyzing football (soccer) team performance using **pandas** and visualizing trends with **matplotlib**.

## Features

- Loads Arsenal FC season-aggregate data from the SportsData.io TeamSeasonStats endpoint into a pandas DataFrame.
- Calculates core team statistics:
  - Matches played
  - Goals scored/conceded
  - Goal difference
  - Average possession
  - Shot accuracy
  - Goals per game
  - Clean sheets
- Generates a human-readable text report.
- Plots performance trends over time (goals and possession).

## Project Structure

```text
.
├── team_analyser.py   # Main script containing TeamAnalyzer
├── Python.gitignore
└── JetBrains.gitIgnore
```

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib

Install dependencies:

```bash
pip install pandas numpy matplotlib
```

Environment configuration (either shell vars or `.env`):

```bash
export SPORTSDATA_API_KEY="<your_sportsdata_key>"
# Optional override; default is UEFA Champions League TeamSeasonStats for Arsenal
export SPORTSDATA_MATCHES_URL="https://api.sportsdata.io/v4/soccer/scores/json/TeamSeasonStats/3/2025"
```

Example `.env` file:

```dotenv
SPORTSDATA_API_KEY=1527a55559834d689d6e2ad76e950fb4
SPORTSDATA_MATCHES_URL=https://api.sportsdata.io/v4/soccer/scores/json/TeamSeasonStats/3/2025
```

## Quick Start

Run the script directly:

```bash
python team_analyser.py
```

What it does:

1. Instantiates `TeamAnalyzer`
2. Loads TeamSeasonStats season totals (all teams in season) and filters to Arsenal FC only
3. Prints a performance report for Arsenal
4. Generates a performance trend figure

> Note: `plot_performance_trends` returns a matplotlib figure object. If you want to display it interactively, call `plt.show()` after invoking the method.

## Usage Example

```python
from team_analyser import TeamAnalyzer
import matplotlib.pyplot as plt

analyzer = TeamAnalyzer()
analyzer.load_sample_data()

report = analyzer.generate_report("Arsenal FC")
print(report)

fig = analyzer.plot_performance_trends("Arsenal FC")
plt.show()
```

## Core API

### `TeamAnalyzer.load_sample_data()`
Loads Arsenal FC season totals from SportsData.io TeamSeasonStats when configured (the loader also reads `.env`). The TeamSeasonStats response contains season aggregates for all teams, and the loader explicitly filters to Arsenal FC rows only. If API loading fails, it falls back to in-repo sample data.

### `TeamAnalyzer.load_match_data_from_api(api_url, api_key, team_name="Arsenal")`
Fetches and parses TeamSeasonStats data from SportsData.io using `Ocp-Apim-Subscription-Key`, then keeps only Arsenal FC season rows.

### `TeamAnalyzer.calculate_basic_stats(team_name)`
Returns a pandas `Series` with aggregated statistics for the given team.

### `TeamAnalyzer.plot_performance_trends(team_name)`
Returns a matplotlib figure with:
- Goals scored vs conceded over time
- Possession over time

### `TeamAnalyzer.generate_report(team_name)`
Builds and returns a formatted text report based on computed statistics.

## Notes

- Example execution is already guarded under `if __name__ == "__main__":`.

## License

No license file is currently included in this repository.
