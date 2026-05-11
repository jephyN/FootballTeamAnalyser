"""Tests for chart generation."""

from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

matplotlib.use('Agg')


class TestPlotPerformanceTrends:
    """Tests for TeamAnalyzer.plot_performance_trends."""

    def test_returns_figure(self, loaded_analyzer):
        fig = loaded_analyzer.plot_performance_trends('Arsenal FC', seasons=(2025, 2026))
        assert fig is not None
        plt.close(fig)

    def test_figure_has_two_axes(self, loaded_analyzer):
        fig = loaded_analyzer.plot_performance_trends('Arsenal FC', seasons=(2025, 2026))
        subplots = [ax for ax in fig.axes if ax.get_subplotspec() is not None]
        assert len(subplots) == 2
        plt.close(fig)

    def test_suptitle_contains_team_name(self, loaded_analyzer):
        fig = loaded_analyzer.plot_performance_trends('Arsenal FC')
        assert 'Arsenal FC' in fig.texts[0].get_text()
        plt.close(fig)

    def test_logo_axes_added_when_logo_present(self, loaded_analyzer):
        fake_logo = Image.new('RGBA', (52, 52), color=(255, 0, 0, 255))
        with patch('team_analyser._fetch_logo_pil', return_value=fake_logo):
            fig = loaded_analyzer.plot_performance_trends(
                'Arsenal FC', logo_url='https://example.com/logo.png'
            )
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_no_logo_axes_without_logo(self, loaded_analyzer):
        with patch('team_analyser._fetch_logo_pil', return_value=None):
            fig = loaded_analyzer.plot_performance_trends(
                'Arsenal FC', logo_url='https://example.com/logo.png'
            )
        assert len(fig.axes) == 2
        plt.close(fig)
