"""Model-level possession prediction logic."""

import importlib
import importlib.util

import numpy as np
import pandas as pd


class _NumpyLinearRegression:
    """Minimal LinearRegression fallback when scikit-learn is unavailable."""

    def __init__(self):
        self.coef_ = None

    def fit(self, x, y):
        ones = np.ones((x.shape[0], 1))
        design = np.hstack((ones, x))
        self.coef_, *_ = np.linalg.lstsq(design, y, rcond=None)

    def predict(self, x):
        ones = np.ones((x.shape[0], 1))
        design = np.hstack((ones, x))
        return design @ self.coef_


def _build_regressor():
    """Return sklearn LinearRegression when available, else numpy fallback."""
    if importlib.util.find_spec('sklearn'):
        sklearn_linear = importlib.import_module('sklearn.linear_model')
        return sklearn_linear.LinearRegression()
    return _NumpyLinearRegression()


class PossessionPredictor:
    """Predict team-vs-opponent possession with Team1 + Team2 = 100%."""

    def __init__(self):
        self._model = _build_regressor()
        self._is_fitted = False

    def fit(self, team_possessions: dict[str, float]):
        """Fit a regression model from known team average possessions."""
        clean = {team: float(val) for team, val in team_possessions.items() if pd.notna(val)}
        if len(clean) < 2:
            raise ValueError('At least two teams with possession values are required.')

        samples = []
        targets = []
        teams = list(clean.items())
        for idx, (_, team_poss) in enumerate(teams):
            for jdx, (_, opp_poss) in enumerate(teams):
                if idx == jdx:
                    continue
                total = team_poss + opp_poss
                if total <= 0:
                    continue
                samples.append([team_poss, opp_poss])
                targets.append((team_poss / total) * 100.0)

        if not samples:
            raise ValueError('Could not build training data for possession model.')

        self._model.fit(np.array(samples), np.array(targets))
        self._is_fitted = True

    def predict_matchup(self, team_possession: float, opponent_possession: float):
        """Predict possession for a matchup and return (team_pct, opp_pct)."""
        if not self._is_fitted:
            raise ValueError('Model is not fitted. Call fit() first.')
        raw_team = float(self._model.predict(np.array([[team_possession, opponent_possession]]))[0])
        team_pct = max(0.0, min(100.0, raw_team))
        return round(team_pct, 2), round(100.0 - team_pct, 2)


def build_predictions(team_name: str, opponents: list[str], team_possessions: dict[str, float]):
    """Build a DataFrame of predicted possession splits vs each opponent."""
    if team_name not in team_possessions:
        raise ValueError(f'No possession baseline found for team: {team_name}')

    predictor = PossessionPredictor()
    predictor.fit(team_possessions)

    rows = []
    team_base = float(team_possessions[team_name])
    for opponent in opponents:
        if opponent == team_name or opponent not in team_possessions:
            continue
        opp_base = float(team_possessions[opponent])
        team_pct, opp_pct = predictor.predict_matchup(team_base, opp_base)
        rows.append({
            'team': team_name,
            'opponent': opponent,
            'team_possession_pred': team_pct,
            'opponent_possession_pred': opp_pct,
        })

    return pd.DataFrame(rows)
