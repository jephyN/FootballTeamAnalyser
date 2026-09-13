"""Tests for possession prediction module."""

import pytest

from possession_prediction import PossessionPredictor, build_predictions


def test_predictor_outputs_sum_to_100():
    predictor = PossessionPredictor()
    predictor.fit({'Arsenal FC': 58.0, 'Chelsea FC': 52.0, 'Barcelona': 61.0})

    team_pct, opp_pct = predictor.predict_matchup(58.0, 52.0)

    assert team_pct + opp_pct == pytest.approx(100.0, abs=0.01)


def test_build_predictions_uses_requested_opponents_only():
    df = build_predictions(
        'Arsenal FC',
        ['Chelsea FC', 'Liverpool FC'],
        {'Arsenal FC': 58.0, 'Chelsea FC': 52.0, 'Liverpool FC': 54.0, 'PSG': 59.0},
    )

    assert list(df['opponent']) == ['Chelsea FC', 'Liverpool FC']
    assert all(
        abs((row.team_possession_pred + row.opponent_possession_pred) - 100.0) < 0.01
        for row in df.itertuples()
    )
