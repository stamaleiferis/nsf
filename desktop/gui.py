"""PyQt5 GUI — all visualization and interaction, delegates computation to processing.py."""

from __future__ import annotations

import sys
from pathlib import Path
from functools import partial

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QPushButton, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QSlider, QProgressBar, QSplitter,
    QScrollArea, QSizePolicy, QStatusBar, QListWidget, QAbstractItemView,
    QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from desktop.processing import (
    generate_dataset, run_separation, run_comparison, run_bp_pipeline,
    get_beat_morphology, run_sweep, run_realtime,
    METHODS, SWEEP_DEFAULTS, SeparationResult, SweepResult, RealtimeResult, BPResult,
)
from src.synth.pulse import pulse_waveform, PulseConfig
from src.data.markers import SyntheticDataset


# ---------------------------------------------------------------------------
# Worker threads
# ---------------------------------------------------------------------------

class WorkerThread(QThread):
    """Generic worker that runs a callable in a background thread."""
    finished = pyqtSignal(object)
    progress = pyqtSignal(int, int)
    error = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._func(*self._args, **self._kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Reusable canvas widget
# ---------------------------------------------------------------------------

class MplCanvas(FigureCanvas):
    def __init__(self, nrows=1, ncols=1, figsize=(8, 4), parent=None):
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = np.array([self.axes])
        self.axes = np.atleast_1d(self.axes).ravel()
        self.fig.set_tight_layout(True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def clear_all(self):
        for ax in self.axes:
            ax.clear()

    def redraw(self):
        self.fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Helper: make a labeled spin box
# ---------------------------------------------------------------------------

def _dspin(label, lo, hi, val, step, decimals=2, parent=None):
    box = QDoubleSpinBox(parent)
    box.setRange(lo, hi)
    box.setValue(val)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    box.setPrefix(f"{label}: ")
    return box


def _ispin(label, lo, hi, val, parent=None):
    box = QSpinBox(parent)
    box.setRange(lo, hi)
    box.setValue(val)
    box.setPrefix(f"{label}: ")
    return box


# ---------------------------------------------------------------------------
# Tab 1: Data Generator
# ---------------------------------------------------------------------------

class GeneratorTab(QWidget):
    dataset_changed = pyqtSignal(object)  # emits SyntheticDataset

    def __init__(self):
        super().__init__()
        self._ds = None
        self._build_ui()

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        # --- Left: controls ---
        ctrl = QWidget()
        cl = QVBoxLayout(ctrl)

        # Grid
        g = QGroupBox("Grid & Timing")
        gl = QGridLayout(g)
        self.sp_rows = _ispin("Rows", 5, 30, 19)
        self.sp_cols = _ispin("Cols", 5, 20, 14)
        self.sp_frames = _ispin("Frames", 60, 900, 300)
        self.sp_fps = _dspin("FPS", 15, 120, 30, 5, 1)
        self.sp_spacing = _dspin("Spacing mm", 1, 4, 2, 0.5)
        for i, w in enumerate([self.sp_rows, self.sp_cols, self.sp_frames, self.sp_fps, self.sp_spacing]):
            gl.addWidget(w, i // 2, i % 2)
        cl.addWidget(g)

        # Pulse
        g2 = QGroupBox("Pulse Model")
        gl2 = QGridLayout(g2)
        self.sp_hr = _dspin("HR bpm", 40, 140, 72, 1, 0)
        self.sp_pamp = _dspin("Amp mm", 0.01, 1, 0.15, 0.01)
        self.sp_cx = _dspin("Center X", -13, 13, 0, 0.5)
        self.sp_cy = _dspin("Center Y", -18, 18, 0, 0.5)
        self.sp_angle = _dspin("Angle deg", -30, 30, 0, 1, 0)
        self.sp_sigma = _dspin("Sigma cross", 1, 8, 3, 0.5)
        self.sp_sigma_along = _dspin("Sigma along", 2, 20, 8, 0.5)
        self.sp_shear = _dspin("Shear ratio", 0, 1, 0.3, 0.05)
        self.sp_tilt = _dspin("Cam tilt", 10, 70, 40, 5, 0)
        for i, w in enumerate([self.sp_hr, self.sp_pamp, self.sp_cx, self.sp_cy,
                                self.sp_angle, self.sp_sigma, self.sp_sigma_along,
                                self.sp_shear, self.sp_tilt]):
            gl2.addWidget(w, i // 2, i % 2)
        cl.addWidget(g2)

        # Artifact
        g3 = QGroupBox("Artifact Model")
        gl3 = QGridLayout(g3)
        self.sp_adeg = _ispin("Degree", 1, 3, 2)
        self.sp_aamp = _dspin("Amp mm", 0, 5, 1, 0.1)
        self.sp_afreq = _dspin("Max freq Hz", 0.1, 5, 2, 0.1)
        self.sp_aseed = _ispin("Seed", 0, 9999, 42)
        for i, w in enumerate([self.sp_adeg, self.sp_aamp, self.sp_afreq, self.sp_aseed]):
            gl3.addWidget(w, i // 2, i % 2)
        cl.addWidget(g3)

        # Noise
        g4 = QGroupBox("Noise")
        gl4 = QHBoxLayout(g4)
        self.sp_nsigma = _dspin("Sigma mm", 0, 0.2, 0.02, 0.005, 3)
        self.sp_nseed = _ispin("Seed", 0, 9999, 43)
        gl4.addWidget(self.sp_nsigma)
        gl4.addWidget(self.sp_nseed)
        cl.addWidget(g4)

        # Marker selector
        g5 = QGroupBox("Marker Inspector")
        gl5 = QHBoxLayout(g5)
        self.sp_mr = _ispin("Row", 0, 18, 9)
        self.sp_mc = _ispin("Col", 0, 13, 7)
        gl5.addWidget(self.sp_mr)
        gl5.addWidget(self.sp_mc)
        cl.addWidget(g5)

        # Frame slider for 2D view
        g6 = QGroupBox("2D Frame")
        gl6 = QHBoxLayout(g6)
        self.sl_frame = QSlider(Qt.Horizontal)
        self.sl_frame.setRange(0, 299)
        self.sl_frame.setValue(50)
        self.lbl_frame = QLabel("50")
        self.sl_frame.valueChanged.connect(lambda v: self.lbl_frame.setText(str(v)))
        gl6.addWidget(self.sl_frame)
        gl6.addWidget(self.lbl_frame)
        cl.addWidget(g6)

        self.btn_gen = QPushButton("Generate")
        self.btn_gen.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_gen.clicked.connect(self._on_generate)
        cl.addWidget(self.btn_gen)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        cl.addWidget(self.progress)

        cl.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(ctrl)
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(340)
        splitter.addWidget(scroll)

        # --- Right: plots ---
        plot_w = QWidget()
        pl = QVBoxLayout(plot_w)

        self.canvas_top = MplCanvas(1, 2, figsize=(10, 3))
        pl.addWidget(self.canvas_top)

        self.canvas_components = MplCanvas(4, 1, figsize=(10, 5))
        pl.addWidget(self.canvas_components)

        self.canvas_2d = MplCanvas(1, 4, figsize=(12, 3))
        pl.addWidget(self.canvas_2d)

        self.lbl_info = QLabel("")
        pl.addWidget(self.lbl_info)

        splitter.addWidget(plot_w)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    def _on_generate(self):
        self.btn_gen.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # indeterminate

        self._worker = WorkerThread(
            generate_dataset,
            num_rows=self.sp_rows.value(), num_cols=self.sp_cols.value(),
            num_frames=self.sp_frames.value(), fps=self.sp_fps.value(),
            spacing=self.sp_spacing.value(),
            heart_rate=self.sp_hr.value(), pulse_amp=self.sp_pamp.value(),
            artery_cx=self.sp_cx.value(), artery_cy=self.sp_cy.value(),
            artery_angle=self.sp_angle.value(), sigma_cross=self.sp_sigma.value(),
            sigma_along=self.sp_sigma_along.value(), shear_ratio=self.sp_shear.value(),
            camera_tilt=self.sp_tilt.value(),
            artifact_degree=self.sp_adeg.value(), artifact_amp=self.sp_aamp.value(),
            artifact_freq=self.sp_afreq.value(), artifact_seed=self.sp_aseed.value(),
            noise_sigma=self.sp_nsigma.value(), noise_seed=self.sp_nseed.value(),
        )
        self._worker.finished.connect(self._on_generated)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_generated(self, ds: SyntheticDataset):
        self._ds = ds
        self.btn_gen.setEnabled(True)
        self.progress.setVisible(False)
        self.sl_frame.setRange(0, ds.markers.num_frames - 1)
        self.sp_mr.setRange(0, ds.markers.num_rows - 1)
        self.sp_mc.setRange(0, ds.markers.num_cols - 1)

        # Set marker to artery center
        peak = np.unravel_index(np.argmax(ds.ground_truth.artery_mask),
                                ds.ground_truth.artery_mask.shape)
        self.sp_mr.setValue(int(peak[0]))
        self.sp_mc.setValue(int(peak[1]))

        self._plot(ds)
        self.dataset_changed.emit(ds)

    def _on_error(self, msg):
        self.btn_gen.setEnabled(True)
        self.progress.setVisible(False)
        self.lbl_info.setText(f"Error: {msg}")

    def _plot(self, ds: SyntheticDataset):
        gt = ds.ground_truth
        gx, gy = gt.rest_positions[..., 0], gt.rest_positions[..., 1]
        t = np.arange(ds.markers.num_frames) / ds.markers.fps

        # Top row: waveform + artery mask
        self.canvas_top.clear_all()
        ax_wf, ax_mask = self.canvas_top.axes

        wf = pulse_waveform(t, PulseConfig(
            heart_rate_bpm=self.sp_hr.value(), amplitude_mm=self.sp_pamp.value()))
        ax_wf.plot(t, wf, 'r-', linewidth=1.5)
        ax_wf.set_title("Pulse Waveform")
        ax_wf.set_xlabel("Time (s)")
        ax_wf.set_ylabel("Amplitude")

        im = ax_mask.imshow(gt.artery_mask, extent=[gx.min(), gx.max(), gy.max(), gy.min()],
                            cmap='hot', vmin=0, vmax=1, aspect='auto')
        ax_mask.set_title("Artery Mask")
        ax_mask.set_xlabel("X (mm)")
        ax_mask.set_ylabel("Y (mm)")
        self.canvas_top.fig.colorbar(im, ax=ax_mask, fraction=0.046)
        self.canvas_top.redraw()

        # Signal components
        mr, mc = self.sp_mr.value(), self.sp_mc.value()
        mr = min(mr, ds.markers.num_rows - 1)
        mc = min(mc, ds.markers.num_cols - 1)

        self.canvas_components.clear_all()
        labels = ["Pulse", "Artifact", "Noise", "Total"]
        data = [
            gt.pulse_displacement[:, mr, mc, 1],
            gt.artifact_displacement[:, mr, mc, 1],
            gt.noise[:, mr, mc, 1],
            (gt.pulse_displacement + gt.artifact_displacement + gt.noise)[:, mr, mc, 1],
        ]
        colors = ['red', 'blue', 'gray', 'black']
        for ax, d, lbl, c in zip(self.canvas_components.axes, data, labels, colors):
            ax.plot(t, d, color=c, linewidth=0.8)
            ax.set_ylabel(f"{lbl} (mm)")
        self.canvas_components.axes[-1].set_xlabel("Time (s)")
        self.canvas_components.axes[0].set_title(
            f"Signal Components at ({mr},{mc}) mask={gt.artery_mask[mr, mc]:.3f}")
        self.canvas_components.redraw()

        # 2D spatial
        frame = self.sl_frame.value()
        frame = min(frame, ds.markers.num_frames - 1)
        self.canvas_2d.clear_all()
        comp_data = [
            gt.pulse_displacement[frame, ..., 1],
            gt.artifact_displacement[frame, ..., 1],
            gt.noise[frame, ..., 1],
            (gt.pulse_displacement + gt.artifact_displacement + gt.noise)[frame, ..., 1],
        ]
        comp_names = ["Pulse", "Artifact", "Noise", "Total"]
        for ax, d, name in zip(self.canvas_2d.axes, comp_data, comp_names):
            vmax = max(np.max(np.abs(d)), 1e-10)
            ax.imshow(d, extent=[gx.min(), gx.max(), gy.max(), gy.min()],
                      cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
            ax.set_title(f"{name} Y-disp")
            ax.set_xlabel("X (mm)")
        self.canvas_2d.axes[0].set_ylabel("Y (mm)")
        self.canvas_2d.redraw()

        # Info
        self.lbl_info.setText(
            f"Grid: {ds.markers.grid_shape}, Frames: {ds.markers.num_frames}, "
            f"FPS: {ds.markers.fps:.0f}, Duration: {ds.markers.duration_sec:.1f}s | "
            f"Pulse RMS: {np.sqrt(np.mean(gt.pulse_displacement**2)):.4f} mm, "
            f"Artifact RMS: {np.sqrt(np.mean(gt.artifact_displacement**2)):.4f} mm, "
            f"Input SNR: {ds.separation_snr():.1f} dB"
        )

    def get_dataset(self) -> SyntheticDataset | None:
        return self._ds


# ---------------------------------------------------------------------------
# Tab 2: Signal Separation
# ---------------------------------------------------------------------------

class SeparationTab(QWidget):
    def __init__(self, get_ds):
        super().__init__()
        self._get_ds = get_ds
        self._result: SeparationResult | None = None
        self._build_ui()

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        # Controls
        ctrl = QWidget()
        cl = QVBoxLayout(ctrl)

        g = QGroupBox("Method")
        gl = QVBoxLayout(g)
        self.cb_method = QComboBox()
        self.cb_method.addItems(METHODS)
        gl.addWidget(self.cb_method)
        cl.addWidget(g)

        g2 = QGroupBox("Parameters")
        gl2 = QGridLayout(g2)
        self.sp_deg = _ispin("Poly degree", 1, 3, 2)
        self.sp_reg = _dspin("Regularization", 1e-9, 0.01, 1e-6, 1e-7, 9)
        self.chk_prefilter = QCheckBox("Temporal pre-filter")
        self.chk_gaussian = QCheckBox("Gaussian extraction")
        self.chk_gaussian.setChecked(True)
        self.sp_niter = _ispin("Iterations", 1, 5, 1)
        self.sp_ncomp = _ispin("Components", 2, 50, 10)
        self.sp_thresh = _dspin("Mask thresh", 0, 1, 0.3, 0.05)
        for i, w in enumerate([self.sp_deg, self.sp_reg, self.chk_prefilter,
                                self.chk_gaussian, self.sp_niter, self.sp_ncomp, self.sp_thresh]):
            gl2.addWidget(w, i, 0)
        cl.addWidget(g2)

        # Marker selector
        g3 = QGroupBox("Marker")
        gl3 = QHBoxLayout(g3)
        self.cb_preset = QComboBox()
        self.cb_preset.addItems(["Artery Center", "Artery Edge", "Off-Artery", "Custom"])
        self.sp_mr = _ispin("Row", 0, 18, 9)
        self.sp_mc = _ispin("Col", 0, 13, 7)
        gl3.addWidget(self.cb_preset)
        gl3.addWidget(self.sp_mr)
        gl3.addWidget(self.sp_mc)
        cl.addWidget(g3)

        # Frame slider for 2D
        g4 = QGroupBox("2D Frame")
        gl4 = QHBoxLayout(g4)
        self.sl_frame = QSlider(Qt.Horizontal)
        self.sl_frame.setRange(0, 299)
        self.sl_frame.setValue(50)
        self.lbl_frame = QLabel("50")
        self.sl_frame.valueChanged.connect(lambda v: self.lbl_frame.setText(str(v)))
        self.sl_frame.valueChanged.connect(self._on_frame_changed)
        gl4.addWidget(self.sl_frame)
        gl4.addWidget(self.lbl_frame)
        cl.addWidget(g4)

        self.btn_run = QPushButton("Run Separation")
        self.btn_run.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_run.clicked.connect(self._on_run)
        cl.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        cl.addWidget(self.progress)

        self.lbl_metrics = QLabel("")
        self.lbl_metrics.setWordWrap(True)
        cl.addWidget(self.lbl_metrics)
        cl.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(ctrl)
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(320)
        splitter.addWidget(scroll)

        # Plots
        plot_w = QWidget()
        pl = QVBoxLayout(plot_w)

        self.canvas_1d = MplCanvas(3, 1, figsize=(10, 6))
        pl.addWidget(self.canvas_1d)

        self.canvas_2d = MplCanvas(2, 3, figsize=(12, 5))
        pl.addWidget(self.canvas_2d)

        splitter.addWidget(plot_w)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    def _on_run(self):
        ds = self._get_ds()
        if ds is None:
            self.lbl_metrics.setText("Generate data first!")
            return

        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)

        method = self.cb_method.currentText()
        self._worker = WorkerThread(
            run_separation, ds, method=method,
            poly_degree=self.sp_deg.value(),
            regularization=self.sp_reg.value(),
            use_prefilter=self.chk_prefilter.isChecked(),
            use_gaussian=self.chk_gaussian.isChecked(),
            n_iterations=self.sp_niter.value(),
            n_components=self.sp_ncomp.value(),
            mask_threshold=self.sp_thresh.value(),
        )
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, result: SeparationResult):
        self._result = result
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)

        m = result.metrics
        self.lbl_metrics.setText(
            f"<b>{result.method}</b><br>"
            f"SNR Improvement: {result.snr_improvement:.1f} dB<br>"
            f"Waveform Corr: {m.waveform_correlation:.3f}<br>"
            f"Spatial Corr: {m.spatial_correlation:.3f}<br>"
            f"Artifact Residual: {m.artifact_residual_fraction:.4f}"
        )

        ds = self._get_ds()
        self.sl_frame.setRange(0, ds.markers.num_frames - 1)
        self._plot_1d(ds, result)
        self._plot_2d(ds, result)

    def _on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        self.lbl_metrics.setText(f"Error: {msg}")

    def _get_marker(self, ds):
        gt = ds.ground_truth
        peak = np.unravel_index(np.argmax(gt.artery_mask), gt.artery_mask.shape)
        preset = self.cb_preset.currentText()
        if preset == "Artery Center":
            return int(peak[0]), int(peak[1])
        elif preset == "Artery Edge":
            edge = np.argwhere((gt.artery_mask > 0.2) & (gt.artery_mask < 0.5))
            if len(edge) > 0:
                dists = np.abs(edge[:, 0] - peak[0])
                idx = tuple(edge[np.argmin(dists)])
                return int(idx[0]), int(idx[1])
            return int(peak[0]), int(peak[1])
        elif preset == "Off-Artery":
            off_col = np.argmin(gt.artery_mask[peak[0], :])
            return int(peak[0]), int(off_col)
        else:
            return self.sp_mr.value(), self.sp_mc.value()

    def _plot_1d(self, ds, result):
        gt = ds.ground_truth
        t = np.arange(ds.markers.num_frames) / ds.markers.fps
        mr, mc = self._get_marker(ds)

        pulse_rel = gt.pulse_displacement - gt.pulse_displacement[0:1]
        artifact_rel = gt.artifact_displacement - gt.artifact_displacement[0:1]
        raw = ds.markers.displacements_from_rest()

        self.canvas_1d.clear_all()
        ax_raw, ax_pulse, ax_art = self.canvas_1d.axes

        ax_raw.plot(t, raw[:, mr, mc, 1], 'k-', linewidth=0.8)
        ax_raw.set_title(f"Raw Signal ({mr},{mc})")
        ax_raw.set_ylabel("Y (mm)")

        ax_pulse.plot(t, pulse_rel[:, mr, mc, 1], 'r--', linewidth=1.5, alpha=0.7, label="GT")
        ax_pulse.plot(t, result.rec_pulse[:, mr, mc, 1], 'g-', linewidth=1.5, label="Recovered")
        ax_pulse.set_title("Pulse: GT vs Recovered")
        ax_pulse.set_ylabel("Y (mm)")
        ax_pulse.legend(fontsize=8)

        ax_art.plot(t, artifact_rel[:, mr, mc, 1], 'b--', linewidth=1.5, alpha=0.7, label="GT")
        ax_art.plot(t, result.est_artifact[:, mr, mc, 1], color='orange', linewidth=1.5, label="Estimated")
        ax_art.set_title("Artifact: GT vs Estimated")
        ax_art.set_xlabel("Time (s)")
        ax_art.set_ylabel("Y (mm)")
        ax_art.legend(fontsize=8)

        self.canvas_1d.redraw()

    def _plot_2d(self, ds, result):
        gt = ds.ground_truth
        gx, gy = gt.rest_positions[..., 0], gt.rest_positions[..., 1]
        ext = [gx.min(), gx.max(), gy.max(), gy.min()]

        pulse_rel = gt.pulse_displacement - gt.pulse_displacement[0:1]
        artifact_rel = gt.artifact_displacement - gt.artifact_displacement[0:1]
        frame = self.sl_frame.value()
        frame = min(frame, ds.markers.num_frames - 1)

        self.canvas_2d.clear_all()
        axes = self.canvas_2d.axes

        panels = [
            (pulse_rel[frame, ..., 1], "GT Pulse"),
            (result.rec_pulse[frame, ..., 1], "Recovered Pulse"),
            (result.rec_pulse[frame, ..., 1] - pulse_rel[frame, ..., 1], "Pulse Error"),
            (artifact_rel[frame, ..., 1], "GT Artifact"),
            (result.est_artifact[frame, ..., 1], "Est. Artifact"),
            (result.est_artifact[frame, ..., 1] - artifact_rel[frame, ..., 1], "Artifact Error"),
        ]

        for ax, (data, title) in zip(axes, panels):
            vmax = max(np.max(np.abs(data)), 1e-10)
            ax.imshow(data, extent=ext, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
            ax.set_title(title, fontsize=9)

        self.canvas_2d.redraw()

    def _on_frame_changed(self, _):
        ds = self._get_ds()
        if ds and self._result:
            self._plot_2d(ds, self._result)


# ---------------------------------------------------------------------------
# Tab 3: Method Comparison
# ---------------------------------------------------------------------------

class ComparisonTab(QWidget):
    def __init__(self, get_ds):
        super().__init__()
        self._get_ds = get_ds
        self._build_ui()

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        ctrl = QWidget()
        cl = QVBoxLayout(ctrl)

        g = QGroupBox("Methods")
        gl = QVBoxLayout(g)
        self.lst_methods = QListWidget()
        self.lst_methods.setSelectionMode(QAbstractItemView.MultiSelection)
        for m in METHODS:
            self.lst_methods.addItem(m)
        # Select first 3 by default
        for i in range(min(3, self.lst_methods.count())):
            self.lst_methods.item(i).setSelected(True)
        gl.addWidget(self.lst_methods)
        cl.addWidget(g)

        self.btn_run = QPushButton("Run All")
        self.btn_run.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_run.clicked.connect(self._on_run)
        cl.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        cl.addWidget(self.progress)

        self.tbl = QTableWidget()
        self.tbl.setColumnCount(5)
        self.tbl.setHorizontalHeaderLabels(["Method", "SNR Imp (dB)", "WF Corr", "Sp Corr", "Art Resid"])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        cl.addWidget(self.tbl)
        cl.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(ctrl)
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(360)
        splitter.addWidget(scroll)

        plot_w = QWidget()
        pl = QVBoxLayout(plot_w)
        self.canvas_bars = MplCanvas(1, 3, figsize=(12, 3.5))
        pl.addWidget(self.canvas_bars)
        self.canvas_overlay = MplCanvas(1, 1, figsize=(10, 4))
        pl.addWidget(self.canvas_overlay)
        splitter.addWidget(plot_w)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    def _on_run(self):
        ds = self._get_ds()
        if ds is None:
            return
        methods = [item.text() for item in self.lst_methods.selectedItems()]
        if not methods:
            return

        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)

        self._worker = WorkerThread(run_comparison, ds, methods)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, results: dict[str, SeparationResult]):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)

        # Table
        self.tbl.setRowCount(len(results))
        for i, (name, r) in enumerate(results.items()):
            self.tbl.setItem(i, 0, QTableWidgetItem(name))
            self.tbl.setItem(i, 1, QTableWidgetItem(f"{r.snr_improvement:.1f}"))
            self.tbl.setItem(i, 2, QTableWidgetItem(f"{r.metrics.waveform_correlation:.3f}"))
            self.tbl.setItem(i, 3, QTableWidgetItem(f"{r.metrics.spatial_correlation:.3f}"))
            self.tbl.setItem(i, 4, QTableWidgetItem(f"{r.metrics.artifact_residual_fraction:.4f}"))

        # Bar charts
        self.canvas_bars.clear_all()
        methods = list(results.keys())
        colors = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2']
        x = np.arange(len(methods))

        metrics_data = [
            ([r.snr_improvement for r in results.values()], "SNR Imp (dB)"),
            ([r.metrics.waveform_correlation for r in results.values()], "WF Corr"),
            ([r.metrics.artifact_residual_fraction for r in results.values()], "Art Resid"),
        ]
        for ax, (vals, title) in zip(self.canvas_bars.axes, metrics_data):
            ax.bar(x, vals, color=colors[:len(methods)])
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=7)
            ax.set_title(title, fontsize=9)
        self.canvas_bars.redraw()

        # Overlay at artery center
        ds = self._get_ds()
        gt = ds.ground_truth
        peak = np.unravel_index(np.argmax(gt.artery_mask), gt.artery_mask.shape)
        r, c = int(peak[0]), int(peak[1])
        t = np.arange(ds.markers.num_frames) / ds.markers.fps
        pulse_rel = gt.pulse_displacement - gt.pulse_displacement[0:1]

        self.canvas_overlay.clear_all()
        ax = self.canvas_overlay.axes[0]
        ax.plot(t, pulse_rel[:, r, c, 1], 'k--', linewidth=2, label="GT Pulse")
        for i, (name, res) in enumerate(results.items()):
            ax.plot(t, res.rec_pulse[:, r, c, 1], color=colors[i % len(colors)],
                    linewidth=1.5, label=name)
        ax.set_title(f"Recovered Pulse at ({r},{c})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Y displacement (mm)")
        ax.legend(fontsize=7)
        self.canvas_overlay.redraw()

    def _on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)


# ---------------------------------------------------------------------------
# Tab 4: BP Estimation
# ---------------------------------------------------------------------------

class BPTab(QWidget):
    def __init__(self, get_ds):
        super().__init__()
        self._get_ds = get_ds
        self._bp_result: BPResult | None = None
        self._build_ui()

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        ctrl = QWidget()
        cl = QVBoxLayout(ctrl)

        g = QGroupBox("Separation")
        gl = QGridLayout(g)
        self.sp_deg = _ispin("Poly degree", 1, 3, 2)
        self.chk_gauss = QCheckBox("Gaussian extraction")
        self.chk_gauss.setChecked(True)
        gl.addWidget(self.sp_deg, 0, 0)
        gl.addWidget(self.chk_gauss, 1, 0)
        cl.addWidget(g)

        g2 = QGroupBox("Pulse Extraction")
        gl2 = QGridLayout(g2)
        self.sp_bp_lo = _dspin("BP low Hz", 0.1, 2, 0.5, 0.1)
        self.sp_bp_hi = _dspin("BP high Hz", 5, 20, 15, 1, 0)
        gl2.addWidget(self.sp_bp_lo, 0, 0)
        gl2.addWidget(self.sp_bp_hi, 0, 1)
        cl.addWidget(g2)

        # Beat selector
        g3 = QGroupBox("Beat Inspector")
        gl3 = QHBoxLayout(g3)
        self.sp_beat = _ispin("Beat #", 0, 0, 0)
        self.sp_beat.valueChanged.connect(self._on_beat_changed)
        gl3.addWidget(self.sp_beat)
        cl.addWidget(g3)

        self.btn_run = QPushButton("Run Pipeline")
        self.btn_run.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_run.clicked.connect(self._on_run)
        cl.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        cl.addWidget(self.progress)

        self.lbl_metrics = QLabel("")
        self.lbl_metrics.setWordWrap(True)
        cl.addWidget(self.lbl_metrics)

        # Beat table
        self.tbl = QTableWidget()
        self.tbl.setColumnCount(7)
        self.tbl.setHorizontalHeaderLabels([
            "Beat", "Duration", "Systolic", "Dicrotic", "Diastolic", "Aug Idx", "Rise Time"])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        cl.addWidget(self.tbl)
        cl.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(ctrl)
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(380)
        splitter.addWidget(scroll)

        plot_w = QWidget()
        pl = QVBoxLayout(plot_w)
        self.canvas_wf = MplCanvas(1, 1, figsize=(10, 3.5))
        pl.addWidget(self.canvas_wf)
        self.canvas_beat = MplCanvas(1, 1, figsize=(10, 3.5))
        pl.addWidget(self.canvas_beat)
        splitter.addWidget(plot_w)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    def _on_run(self):
        ds = self._get_ds()
        if ds is None:
            return

        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)

        self._worker = WorkerThread(
            run_bp_pipeline, ds,
            poly_degree=self.sp_deg.value(),
            use_gaussian=self.chk_gauss.isChecked(),
            bp_low=self.sp_bp_lo.value(),
            bp_high=self.sp_bp_hi.value(),
        )
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, result: BPResult):
        self._bp_result = result
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)

        ds = self._get_ds()
        timing_str = ", ".join(f"{k}: {v:.1f}ms" for k, v in result.timing.items())
        sys_str = f"{result.systolic:.0f} mmHg" if result.systolic > 0 else "N/A"
        dia_str = f"{result.diastolic:.0f} mmHg" if result.diastolic > 0 else "N/A"
        self.lbl_metrics.setText(
            f"<b>HR:</b> {result.heart_rate:.0f} BPM | "
            f"<b>Confidence:</b> {result.confidence:.2f}<br>"
            f"<b>Systolic:</b> {sys_str} | <b>Diastolic:</b> {dia_str}<br>"
            f"<b>Timing:</b> {timing_str}"
        )

        # Waveform plot
        wf = result.waveform
        t = np.arange(len(wf)) / ds.markers.fps

        self.canvas_wf.clear_all()
        ax = self.canvas_wf.axes[0]
        ax.plot(t, wf, 'b-', linewidth=1.5)
        for s, e in result.beats:
            ax.axvline(t[s], color='green', linestyle=':', alpha=0.4)
        ax.set_title(f"Extracted Pulse ({len(result.beats)} beats)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        self.canvas_wf.redraw()

        # Beat selector
        if result.beats:
            self.sp_beat.setRange(0, len(result.beats) - 1)
            self.sp_beat.setValue(len(result.beats) // 2)
            self._plot_beat(result, ds.markers.fps)

        # Beat table
        self.tbl.setRowCount(len(result.beats))
        for i, (s, e) in enumerate(result.beats):
            morph = get_beat_morphology(wf, s, e, ds.markers.fps)
            self.tbl.setItem(i, 0, QTableWidgetItem(str(i)))
            self.tbl.setItem(i, 1, QTableWidgetItem(f"{morph.beat_duration:.3f}"))
            self.tbl.setItem(i, 2, QTableWidgetItem(f"{morph.systolic_peak_value:.4f}"))
            self.tbl.setItem(i, 3, QTableWidgetItem(f"{morph.dicrotic_notch_value:.4f}"))
            self.tbl.setItem(i, 4, QTableWidgetItem(f"{morph.diastolic_peak_value:.4f}"))
            self.tbl.setItem(i, 5, QTableWidgetItem(f"{morph.augmentation_index:.3f}"))
            self.tbl.setItem(i, 6, QTableWidgetItem(f"{morph.rise_time:.4f}"))

    def _on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        self.lbl_metrics.setText(f"Error: {msg}")

    def _on_beat_changed(self, _):
        if self._bp_result and self._bp_result.beats:
            ds = self._get_ds()
            if ds:
                self._plot_beat(self._bp_result, ds.markers.fps)

    def _plot_beat(self, result, fps):
        idx = self.sp_beat.value()
        if idx >= len(result.beats):
            return
        s, e = result.beats[idx]
        morph = get_beat_morphology(result.waveform, s, e, fps)
        t_beat = np.arange(e - s) / fps
        wf_beat = result.waveform[s:e]

        self.canvas_beat.clear_all()
        ax = self.canvas_beat.axes[0]
        ax.plot(t_beat, wf_beat, 'b-', linewidth=2)
        ax.plot(morph.systolic_peak_time, morph.systolic_peak_value, 'rv', markersize=12,
                label=f"Systolic ({morph.systolic_peak_value:.3f})")
        ax.plot(morph.dicrotic_notch_time, morph.dicrotic_notch_value, 'g^', markersize=10,
                label=f"Dicrotic ({morph.dicrotic_notch_value:.3f})")
        ax.plot(morph.diastolic_peak_time, morph.diastolic_peak_value, 'ms', markersize=10,
                label=f"Diastolic ({morph.diastolic_peak_value:.3f})")
        ax.set_title(f"Beat {idx} (dur={morph.beat_duration:.3f}s, AI={morph.augmentation_index:.2f})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend(fontsize=7)
        self.canvas_beat.redraw()


# ---------------------------------------------------------------------------
# Tab 5: Parameter Sweep
# ---------------------------------------------------------------------------

class SweepTab(QWidget):
    def __init__(self, get_ds):
        super().__init__()
        self._get_ds = get_ds
        self._build_ui()

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        ctrl = QWidget()
        cl = QVBoxLayout(ctrl)

        g = QGroupBox("Sweep Config")
        gl = QGridLayout(g)
        self.cb_param = QComboBox()
        self.cb_param.addItems(list(SWEEP_DEFAULTS.keys()))
        self.cb_param.currentTextChanged.connect(self._on_param_changed)
        gl.addWidget(QLabel("Parameter:"), 0, 0)
        gl.addWidget(self.cb_param, 0, 1)

        self.sp_min = _dspin("Min", -100, 100, 0.01, 0.01, 3)
        self.sp_max = _dspin("Max", -100, 100, 0.5, 0.01, 3)
        self.sp_npts = _ispin("Points", 2, 20, 8)
        self.sp_sep_deg = _ispin("Sep degree", 1, 3, 2)
        gl.addWidget(self.sp_min, 1, 0)
        gl.addWidget(self.sp_max, 1, 1)
        gl.addWidget(self.sp_npts, 2, 0)
        gl.addWidget(self.sp_sep_deg, 2, 1)
        cl.addWidget(g)

        self.btn_run = QPushButton("Run Sweep")
        self.btn_run.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_run.clicked.connect(self._on_run)
        cl.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        cl.addWidget(self.progress)
        cl.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(ctrl)
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(320)
        splitter.addWidget(scroll)

        plot_w = QWidget()
        pl = QVBoxLayout(plot_w)
        self.canvas_sweep = MplCanvas(1, 1, figsize=(10, 4))
        pl.addWidget(self.canvas_sweep)
        self.canvas_residual = MplCanvas(1, 1, figsize=(10, 3.5))
        pl.addWidget(self.canvas_residual)
        splitter.addWidget(plot_w)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        # Set defaults for first param
        self._on_param_changed(self.cb_param.currentText())

    def _on_param_changed(self, param):
        if param in SWEEP_DEFAULTS:
            d = SWEEP_DEFAULTS[param]
            self.sp_min.setValue(d[0])
            self.sp_max.setValue(d[1])
            self.sp_npts.setValue(d[2])

    def _on_run(self):
        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        def on_progress(done, total):
            self.progress.setValue(int(done / total * 100))

        self._worker = WorkerThread(
            run_sweep,
            self.cb_param.currentText(),
            self.sp_min.value(), self.sp_max.value(), self.sp_npts.value(),
            sep_degree=self.sp_sep_deg.value(),
            progress_callback=on_progress,
        )
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, result: SweepResult):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)

        # Dual-axis sweep
        self.canvas_sweep.clear_all()
        ax = self.canvas_sweep.axes[0]
        ax2 = ax.twinx()
        ax.plot(result.values, result.snr_improvement, 'b-o', linewidth=1.5, label="SNR Imp (dB)")
        ax2.plot(result.values, result.waveform_correlation, 'r--s', linewidth=1.5, label="WF Corr")
        ax.set_xlabel(result.param_name.replace("_", " "))
        ax.set_ylabel("SNR Improvement (dB)", color='blue')
        ax2.set_ylabel("Waveform Correlation", color='red')
        ax2.set_ylim(-0.1, 1.1)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
        ax.set_title(f"Sweep: {result.param_name}")
        self.canvas_sweep.redraw()

        # Residual bar chart
        self.canvas_residual.clear_all()
        ax = self.canvas_residual.axes[0]
        x = np.arange(len(result.values))
        ax.bar(x, [r * 100 for r in result.artifact_residual], color='steelblue')
        ax.axhline(5.0, color='red', linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.2g}" for v in result.values], rotation=30, fontsize=8)
        ax.set_xlabel(result.param_name.replace("_", " "))
        ax.set_ylabel("%")
        ax.set_title("Artifact Residual (%)")
        self.canvas_residual.redraw()

    def _on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)


# ---------------------------------------------------------------------------
# Tab 6: Real-Time Demo
# ---------------------------------------------------------------------------

class RealtimeTab(QWidget):
    def __init__(self, get_ds):
        super().__init__()
        self._get_ds = get_ds
        self._build_ui()

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        ctrl = QWidget()
        cl = QVBoxLayout(ctrl)

        g = QGroupBox("Config")
        gl = QVBoxLayout(g)
        self.sp_frames = _ispin("Frames", 30, 900, 300)
        gl.addWidget(self.sp_frames)
        cl.addWidget(g)

        self.btn_run = QPushButton("Run Real-Time")
        self.btn_run.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_run.clicked.connect(self._on_run)
        cl.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        cl.addWidget(self.progress)

        self.lbl_stats = QLabel("")
        self.lbl_stats.setWordWrap(True)
        cl.addWidget(self.lbl_stats)
        cl.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(ctrl)
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(300)
        splitter.addWidget(scroll)

        plot_w = QWidget()
        pl = QVBoxLayout(plot_w)
        self.canvas = MplCanvas(1, 2, figsize=(10, 4))
        pl.addWidget(self.canvas)
        splitter.addWidget(plot_w)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    def _on_run(self):
        ds = self._get_ds()
        if ds is None:
            return

        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        n = min(self.sp_frames.value(), ds.markers.num_frames)

        def on_progress(done, total):
            self.progress.setValue(int(done / total * 100))

        self._worker = WorkerThread(run_realtime, ds, n, progress_callback=on_progress)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, result: RealtimeResult):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)

        arr = np.array(result.latencies)
        target = 5.0
        status = "PASS" if np.median(arr) < target else "FAIL"

        self.lbl_stats.setText(
            f"<b>Median:</b> {np.median(arr):.3f} ms | "
            f"<b>Mean:</b> {np.mean(arr):.3f} ms<br>"
            f"<b>P95:</b> {np.percentile(arr, 95):.3f} ms | "
            f"<b>P99:</b> {np.percentile(arr, 99):.3f} ms | "
            f"<b>Max:</b> {np.max(arr):.3f} ms<br>"
            f"<b>Target ({target}ms):</b> {status}<br>"
            f"Grid: {result.grid_shape[0]}x{result.grid_shape[1]} = {result.n_markers} markers | "
            f"Throughput: {len(result.latencies) / (sum(result.latencies) / 1000):.0f} fps"
        )

        self.canvas.clear_all()
        ax_hist, ax_time = self.canvas.axes

        ax_hist.hist(arr, bins=50, color='steelblue', edgecolor='white')
        ax_hist.axvline(np.median(arr), color='red', linestyle='--', label=f"Median {np.median(arr):.2f}")
        ax_hist.axvline(target, color='orange', linestyle=':', label=f"Target {target}")
        ax_hist.set_xlabel("Latency (ms)")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Latency Distribution")
        ax_hist.legend(fontsize=7)

        ax_time.scatter(range(len(arr)), arr, s=2, color='steelblue', alpha=0.5)
        ax_time.axhline(np.median(arr), color='red', linestyle='--')
        ax_time.axhline(target, color='orange', linestyle=':')
        ax_time.set_xlabel("Frame")
        ax_time.set_ylabel("Latency (ms)")
        ax_time.set_title("Latency Over Time")

        self.canvas.redraw()

    def _on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        self.lbl_stats.setText(f"Error: {msg}")


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NSF — Arterial Pulse Tonometry: Signal Separation")
        self.resize(1400, 900)

        tabs = QTabWidget()

        self.gen_tab = GeneratorTab()
        tabs.addTab(self.gen_tab, "Data Generator")

        get_ds = self.gen_tab.get_dataset

        self.sep_tab = SeparationTab(get_ds)
        tabs.addTab(self.sep_tab, "Signal Separation")

        self.cmp_tab = ComparisonTab(get_ds)
        tabs.addTab(self.cmp_tab, "Method Comparison")

        self.bp_tab = BPTab(get_ds)
        tabs.addTab(self.bp_tab, "BP Estimation")

        self.sweep_tab = SweepTab(get_ds)
        tabs.addTab(self.sweep_tab, "Parameter Sweep")

        self.rt_tab = RealtimeTab(get_ds)
        tabs.addTab(self.rt_tab, "Real-Time Demo")

        self.setCentralWidget(tabs)

        status = QStatusBar()
        status.showMessage("Generate data to get started")
        self.setStatusBar(status)

        # Update status when dataset changes
        self.gen_tab.dataset_changed.connect(
            lambda ds: status.showMessage(
                f"Dataset: {ds.markers.grid_shape} grid, {ds.markers.num_frames} frames, "
                f"{ds.markers.fps:.0f} FPS"
            )
        )


def run():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
