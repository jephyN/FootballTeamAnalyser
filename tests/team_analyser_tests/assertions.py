"""Shared assertions for team_analyser tests."""


def assert_row_count(rows, expected_count):
    """Assert a row collection contains the expected number of rows."""
    assert len(rows) == expected_count


def assert_no_rows(rows):
    """Assert a row collection is empty."""
    assert not rows


def assert_season_years(data_frame, expected_years):
    """Assert a DataFrame contains exactly the expected season years."""
    assert set(data_frame['season_year']) == set(expected_years)


def assert_single_season_frame(data_frame, season_year):
    """Assert a one-row DataFrame for the expected season year was returned."""
    assert_row_count(data_frame, 1)
    assert data_frame.iloc[0]['season_year'] == season_year


def assert_default_rows_loaded(analyzer, expected_team=None):
    """Assert fallback sample match_data was loaded."""
    assert analyzer.match_data is not None
    assert_row_count(analyzer.match_data, 2)
    if expected_team is not None:
        assert (analyzer.match_data['team'] == expected_team).all()


def assert_warning_mentions(caught_warnings, text):
    """Assert at least one captured warning contains the given text."""
    messages = [str(warning.message) for warning in caught_warnings]
    assert any(text in message for message in messages)


def assert_contains_text(output, *expected_parts):
    """Assert output contains all expected text fragments."""
    for expected in expected_parts:
        assert expected in output


def assert_sorted_unique_teams(teams, *expected_team_names):
    """Assert team names are sorted, deduplicated, and include expected names."""
    assert teams == sorted(teams)
    assert_row_count(teams, len(set(teams)))
    for team_name in expected_team_names:
        assert team_name in teams
