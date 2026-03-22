"""Plotly figure builders for all visualization types."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_pulse_waveform(t: np.ndarray, waveform: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=waveform, mode="lines", line=dict(color="red", width=2)))
    fig.update_layout(
        title="Pulse Waveform", xaxis_title="Time (s)", yaxis_title="Amplitude",
        height=300, margin=dict(t=40, b=40),
    )
    return fig


def plot_artery_mask(mask: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Artery Mask", "Cross-Section (Y=center)"])

    fig.add_trace(go.Heatmap(
        z=mask, x=grid_x[0, :], y=grid_y[:, 0],
        colorscale="Hot", zmin=0, zmax=1, colorbar=dict(title="Influence", x=0.45),
    ), row=1, col=1)

    mid = mask.shape[0] // 2
    fig.add_trace(go.Scatter(
        x=grid_x[0, :], y=mask[mid, :], mode="lines+markers",
        line=dict(color="red"), marker=dict(size=5),
    ), row=1, col=2)
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=1, col=2)

    fig.update_layout(height=350, margin=dict(t=40, b=40), showlegend=False)
    fig.update_xaxes(title_text="X (mm)", row=1, col=1)
    fig.update_yaxes(title_text="Y (mm)", row=1, col=1)
    fig.update_xaxes(title_text="X (mm)", row=1, col=2)
    fig.update_yaxes(title_text="Mask Value", row=1, col=2)
    return fig


def plot_heatmap(data_2d: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray,
                 title: str = "", colorscale: str = "RdBu_r", symmetric: bool = True) -> go.Figure:
    zmax = np.max(np.abs(data_2d)) if symmetric else None
    zmin = -zmax if symmetric and zmax else None
    fig = go.Figure(go.Heatmap(
        z=data_2d, x=grid_x[0, :], y=grid_y[:, 0],
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        colorbar=dict(title="mm"),
    ))
    fig.update_layout(title=title, xaxis_title="X (mm)", yaxis_title="Y (mm)",
                      height=400, margin=dict(t=40, b=40), yaxis=dict(autorange="reversed"))
    return fig


def plot_quiver(grid_x: np.ndarray, grid_y: np.ndarray, disp: np.ndarray,
                title: str = "") -> go.Figure:
    """Plot displacement vectors using Plotly scatter + annotations."""
    fig = go.Figure()

    # Marker positions
    fig.add_trace(go.Scatter(
        x=grid_x.ravel(), y=grid_y.ravel(), mode="markers",
        marker=dict(size=3, color="gray"), showlegend=False,
    ))

    # Arrows as line segments
    scale = np.max(np.abs(disp)) * 5 if np.max(np.abs(disp)) > 0 else 1.0
    dx = disp[..., 0] / scale * (grid_x.max() - grid_x.min()) * 0.08
    dy = disp[..., 1] / scale * (grid_y.max() - grid_y.min()) * 0.08

    for r in range(grid_x.shape[0]):
        for c in range(grid_x.shape[1]):
            if abs(dx[r, c]) + abs(dy[r, c]) > 1e-8:
                fig.add_annotation(
                    x=grid_x[r, c] + dx[r, c], y=grid_y[r, c] + dy[r, c],
                    ax=grid_x[r, c], ay=grid_y[r, c],
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                    arrowcolor="steelblue",
                )

    fig.update_layout(
        title=title, xaxis_title="X (mm)", yaxis_title="Y (mm)",
        height=450, margin=dict(t=40, b=40),
        yaxis=dict(autorange="reversed", scaleanchor="x"),
    )
    return fig


def plot_signal_components(t: np.ndarray, pulse: np.ndarray, artifact: np.ndarray,
                           noise: np.ndarray, total: np.ndarray,
                           marker_label: str = "") -> go.Figure:
    """4-panel time series for one marker."""
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=["Pulse", "Artifact", "Noise", "Total"])

    fig.add_trace(go.Scatter(x=t, y=pulse, line=dict(color="red", width=1), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=artifact, line=dict(color="blue", width=1), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=noise, line=dict(color="gray", width=0.5), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=total, line=dict(color="black", width=0.8), showlegend=False), row=4, col=1)

    fig.update_layout(height=600, margin=dict(t=40, b=40),
                      title=f"Signal Components — {marker_label}")
    fig.update_xaxes(title_text="Time (s)", row=4, col=1)
    for i in range(1, 5):
        fig.update_yaxes(title_text="Y (mm)", row=i, col=1)
    return fig


def plot_before_after(t: np.ndarray, raw: np.ndarray,
                      recovered: np.ndarray, gt_pulse: np.ndarray,
                      est_artifact: np.ndarray, gt_artifact: np.ndarray,
                      marker_label: str = "") -> go.Figure:
    """Before/after separation: raw, pulse comparison, artifact comparison."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["Raw Signal", "Pulse: GT vs Recovered",
                                        "Artifact: GT vs Estimated"])

    fig.add_trace(go.Scatter(x=t, y=raw, line=dict(color="black", width=0.8),
                             name="Raw"), row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=gt_pulse, line=dict(color="red", width=1.5, dash="dash"),
                             name="GT Pulse", opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=recovered, line=dict(color="green", width=1.5),
                             name="Recovered"), row=2, col=1)

    fig.add_trace(go.Scatter(x=t, y=gt_artifact, line=dict(color="blue", width=1.5, dash="dash"),
                             name="GT Artifact", opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=est_artifact, line=dict(color="orange", width=1.5),
                             name="Estimated"), row=3, col=1)

    fig.update_layout(height=600, margin=dict(t=40, b=40),
                      title=f"Separation Results — {marker_label}")
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    return fig


def plot_pulse_zoom(t: np.ndarray, gt_pulse: np.ndarray, recovered: np.ndarray,
                    marker_label: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=gt_pulse, line=dict(color="red", width=2, dash="dash"),
                             name="GT Pulse", opacity=0.7))
    fig.add_trace(go.Scatter(x=t, y=recovered, line=dict(color="green", width=2),
                             name="Recovered"))
    fig.update_layout(
        title=f"Pulse Zoom — {marker_label}",
        xaxis_title="Time (s)", yaxis_title="Y displacement (mm)",
        height=350, margin=dict(t=40, b=40),
    )
    return fig


def plot_method_comparison_bars(results: dict[str, dict]) -> go.Figure:
    """Bar chart comparing methods across metrics."""
    methods = list(results.keys())
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["SNR Improvement (dB)", "Waveform Corr", "Artifact Residual"])

    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]

    for i, metric in enumerate(["snr_improvement_db", "waveform_correlation", "artifact_residual_fraction"]):
        vals = [results[m].get(metric, 0) for m in methods]
        fig.add_trace(go.Bar(
            x=methods, y=vals, marker_color=colors[:len(methods)],
            showlegend=False,
        ), row=1, col=i + 1)

    fig.update_layout(height=350, margin=dict(t=50, b=40))
    return fig


def plot_beat_morphology(t_beat: np.ndarray, beat_wf: np.ndarray,
                         morph) -> go.Figure:
    """Single beat with morphology annotations."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_beat, y=beat_wf, mode="lines",
                             line=dict(color="blue", width=2), name="Beat"))

    fig.add_trace(go.Scatter(
        x=[morph.systolic_peak_time], y=[morph.systolic_peak_value],
        mode="markers", marker=dict(size=14, symbol="triangle-down", color="red"),
        name=f"Systolic ({morph.systolic_peak_value:.3f})",
    ))
    fig.add_trace(go.Scatter(
        x=[morph.dicrotic_notch_time], y=[morph.dicrotic_notch_value],
        mode="markers", marker=dict(size=12, symbol="triangle-up", color="green"),
        name=f"Dicrotic Notch ({morph.dicrotic_notch_value:.3f})",
    ))
    fig.add_trace(go.Scatter(
        x=[morph.diastolic_peak_time], y=[morph.diastolic_peak_value],
        mode="markers", marker=dict(size=12, symbol="square", color="purple"),
        name=f"Diastolic ({morph.diastolic_peak_value:.3f})",
    ))

    fig.update_layout(
        title=f"Beat Morphology (duration={morph.beat_duration:.3f}s, AI={morph.augmentation_index:.2f})",
        xaxis_title="Time (s)", yaxis_title="Amplitude",
        height=350, margin=dict(t=40, b=40),
    )
    return fig


def plot_latency_histogram(latencies: list[float]) -> go.Figure:
    import numpy as np
    arr = np.array(latencies)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Latency Distribution", "Latency Over Time"])

    fig.add_trace(go.Histogram(x=arr, nbinsx=50, marker_color="steelblue", showlegend=False), row=1, col=1)
    fig.add_vline(x=np.median(arr), line_dash="dash", line_color="red", row=1, col=1)
    fig.add_vline(x=5.0, line_dash="dot", line_color="orange", row=1, col=1)

    fig.add_trace(go.Scatter(y=arr, mode="markers", marker=dict(size=2, color="steelblue", opacity=0.5),
                             showlegend=False), row=1, col=2)
    fig.add_hline(y=np.median(arr), line_dash="dash", line_color="red", row=1, col=2)
    fig.add_hline(y=5.0, line_dash="dot", line_color="orange", row=1, col=2)

    fig.update_layout(height=350, margin=dict(t=50, b=40))
    fig.update_xaxes(title_text="Latency (ms)", row=1, col=1)
    fig.update_xaxes(title_text="Frame", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Latency (ms)", row=1, col=2)
    return fig


def plot_sweep(values: list, metric_values: dict[str, list]) -> go.Figure:
    """Dual-axis sweep plot."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if "snr_improvement_db" in metric_values:
        fig.add_trace(go.Scatter(
            x=values, y=metric_values["snr_improvement_db"],
            mode="lines+markers", name="SNR Improvement (dB)",
            line=dict(color="blue"),
        ), secondary_y=False)

    if "waveform_correlation" in metric_values:
        fig.add_trace(go.Scatter(
            x=values, y=metric_values["waveform_correlation"],
            mode="lines+markers", name="Waveform Corr",
            line=dict(color="red", dash="dash"),
        ), secondary_y=True)

    fig.update_yaxes(title_text="SNR Improvement (dB)", secondary_y=False)
    fig.update_yaxes(title_text="Waveform Correlation", secondary_y=True, range=[-0.1, 1.1])
    fig.update_layout(height=400, margin=dict(t=40, b=40))
    return fig
