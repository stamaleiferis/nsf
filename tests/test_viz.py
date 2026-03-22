"""Smoke tests for visualization modules."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend

from src.synth.generator import GeneratorConfig, generate
from src.viz.plots import (
    plot_marker_grid, plot_displacement_timeseries,
    plot_artery_mask, plot_spatial_snapshot,
)
from src.viz.animate import animate_markers, animate_components


def _make_dataset():
    return generate(GeneratorConfig(num_frames=30, num_rows=5, num_cols=4))


class TestPlots:
    def test_plot_marker_grid(self):
        ds = _make_dataset()
        fig = plot_marker_grid(ds.markers, frame=0)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_displacement_timeseries(self):
        ds = _make_dataset()
        fig = plot_displacement_timeseries(ds, row=2, col=2)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_artery_mask(self):
        ds = _make_dataset()
        fig = plot_artery_mask(ds)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_spatial_snapshot(self):
        ds = _make_dataset()
        fig = plot_spatial_snapshot(ds, frame=0, component="pulse")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestAnimate:
    def test_animate_markers_creates(self):
        ds = _make_dataset()
        anim = animate_markers(ds.markers)
        assert anim is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_animate_components_creates(self):
        ds = _make_dataset()
        anim = animate_components(ds)
        assert anim is not None
        import matplotlib.pyplot as plt
        plt.close("all")
