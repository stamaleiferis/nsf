"""Tests for MarkerTimeSeries data structures."""

import numpy as np
import pytest

from src.data.markers import MarkerTimeSeries, GroundTruth, SyntheticDataset


def _make_markers(T=10, R=5, C=4, fps=30.0):
    positions = np.random.randn(T, R, C, 2)
    visibility = np.ones((T, R, C))
    return MarkerTimeSeries(positions=positions, visibility=visibility, fps=fps)


class TestMarkerTimeSeries:
    def test_shape_properties(self):
        m = _make_markers(T=20, R=5, C=4)
        assert m.num_frames == 20
        assert m.num_rows == 5
        assert m.num_cols == 4
        assert m.grid_shape == (5, 4)

    def test_duration(self):
        m = _make_markers(T=60, fps=30.0)
        assert m.duration_sec == pytest.approx(2.0)

    def test_velocities_shape(self):
        m = _make_markers(T=10, R=5, C=4)
        v = m.velocities
        assert v.shape == (9, 5, 4, 2)

    def test_velocities_values(self):
        m = _make_markers(T=3, R=2, C=2)
        v = m.velocities
        expected = m.positions[1:] - m.positions[:-1]
        np.testing.assert_array_equal(v, expected)

    def test_displacements_from_rest(self):
        m = _make_markers(T=5, R=3, C=3)
        d = m.displacements_from_rest(rest_frame=0)
        assert d.shape == m.positions.shape
        np.testing.assert_array_equal(d[0], 0.0)

    def test_bad_visibility_shape_raises(self):
        with pytest.raises(AssertionError):
            MarkerTimeSeries(
                positions=np.zeros((10, 5, 4, 2)),
                visibility=np.ones((10, 5, 3)),  # wrong C
            )

    def test_bad_xy_dim_raises(self):
        with pytest.raises(AssertionError):
            MarkerTimeSeries(
                positions=np.zeros((10, 5, 4, 3)),  # 3 instead of 2
                visibility=np.ones((10, 5, 4)),
            )


class TestSyntheticDataset:
    def test_separation_snr(self):
        T, R, C = 10, 3, 3
        gt = GroundTruth(
            pulse_displacement=np.ones((T, R, C, 2)),
            artifact_displacement=0.1 * np.ones((T, R, C, 2)),
            noise=np.zeros((T, R, C, 2)),
            artery_mask=np.ones((R, C)),
            pulse_waveform=np.ones(T),
            rest_positions=np.zeros((R, C, 2)),
        )
        m = _make_markers(T=T, R=R, C=C)
        ds = SyntheticDataset(markers=m, ground_truth=gt)
        snr = ds.separation_snr()
        assert snr == pytest.approx(10 * np.log10(1.0 / 0.01), abs=0.01)

    def test_snr_infinite_no_artifact(self):
        T, R, C = 5, 2, 2
        gt = GroundTruth(
            pulse_displacement=np.ones((T, R, C, 2)),
            artifact_displacement=np.zeros((T, R, C, 2)),
            noise=np.zeros((T, R, C, 2)),
            artery_mask=np.ones((R, C)),
            pulse_waveform=np.ones(T),
            rest_positions=np.zeros((R, C, 2)),
        )
        m = _make_markers(T=T, R=R, C=C)
        ds = SyntheticDataset(markers=m, ground_truth=gt)
        assert ds.separation_snr() == float("inf")
