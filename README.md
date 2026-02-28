# FootballTeamAnalyser

A lightweight Python project for analyzing football (soccer) team performance using **pandas** and visualizing trends with **matplotlib**.

## Features

- Loads Arsenal match data from the SportsData.io Soccer API into a pandas DataFrame.
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

Environment configuration:

```bash
export SPORTSDATA_API_KEY="<your_sportsdata_key>"
# Optional override; defaults to GamesByTeam/EPL/ARS
export SPORTSDATA_MATCHES_URL="https://api.sportsdata.io/v4/soccer/scores/json/GamesByTeam/EPL/ARS"
```

## Quick Start

Run the script directly:

```bash
python team_analyser.py
```

What it does:

1. Instantiates `TeamAnalyzer`
2. Loads SportsData.io data using `SPORTSDATA_API_KEY` (and optional `SPORTSDATA_MATCHES_URL`; defaults to `GamesByTeam/EPL/ARS`)
3. Prints a performance report for Arsenal
4. Generates a performance trend figure

> Note: `plot_performance_trends` returns a matplotlib figure object. If you want to display it interactively, call `plt.show()` after invoking the method.

## Usage Example

```python
from team_analyser import TeamAnalyzer
import matplotlib.pyplot as plt

analyzer = TeamAnalyzer()
analyzer.load_sample_data()

report = analyzer.generate_report("Arsenal")
print(report)

fig = analyzer.plot_performance_trends("Arsenal")
plt.show()
```

## Core API

### `TeamAnalyzer.load_sample_data()`
Loads Arsenal matches from SportsData.io when `SPORTSDATA_API_KEY` is configured. It uses `SPORTSDATA_MATCHES_URL` if provided, otherwise defaults to `https://api.sportsdata.io/v4/soccer/scores/json/GamesByTeam/EPL/ARS`. If API loading fails, it falls back to in-repo sample data.

### `TeamAnalyzer.load_match_data_from_api(api_url, api_key, team_name="Arsenal")`
Fetches and parses match data directly from a SportsData.io endpoint using `Ocp-Apim-Subscription-Key`. The parser handles common response field variants (for home/away team names, scores, and match date).

### `TeamAnalyzer.calculate_basic_stats(team_name)`
Returns a pandas `Series` with aggregated statistics for the given team.

### `TeamAnalyzer.plot_performance_trends(team_name)`
Returns a matplotlib figure with:
- Goals scored vs conceded over time
- Possession over time

### `TeamAnalyzer.generate_report(team_name)`
Builds and returns a formatted text report based on computed statistics.

## Notes

- The current script includes executable example code at module level.
- For reuse as a package/module, consider moving example execution under:

```python
if __name__ == "__main__":
    ...
```

## License

No license file is currently included in this repository.
