"""Page 6: Real-Time Demo

Frame-by-frame processing with latency measurement.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import streamlit as st

from src.synth.generator import GeneratorConfig, generate
from src.pipeline import RealTimeSeparator
from webapp.components.plot_builders import plot_latency_histogram
from webapp.components.state import ensure_dataset

st.set_page_config(page_title="Real-Time Demo", layout="wide")
st.title("Real-Time Processing Demo")

ds = ensure_dataset()
if ds is None:
    st.stop()

gt = ds.ground_truth
gx, gy = gt.rest_positions[..., 0], gt.rest_positions[..., 1]

# Sidebar
st.sidebar.header("Config")
n_frames = st.sidebar.slider("Frames to Process", 30, ds.markers.num_frames, min(300, ds.markers.num_frames), 30)

if st.sidebar.button("Run Real-Time", type="primary"):
    rt = RealTimeSeparator(gx, gy, gt.artery_mask)
    latencies = []
    progress = st.progress(0)

    for i in range(n_frames):
        _, lat = rt.process_frame(ds.markers.positions[i])
        latencies.append(lat)
        if i % 30 == 0:
            progress.progress((i + 1) / n_frames)

    progress.progress(1.0)
    st.session_state["rt_latencies"] = latencies

if "rt_latencies" not in st.session_state:
    st.info("Click **Run Real-Time** to measure per-frame latency.")
    st.stop()

latencies = st.session_state["rt_latencies"]
arr = np.array(latencies)

# Stats
st.subheader("Latency Statistics")
cols = st.columns(5)
cols[0].metric("Median", f"{np.median(arr):.3f} ms")
cols[1].metric("Mean", f"{np.mean(arr):.3f} ms")
cols[2].metric("P95", f"{np.percentile(arr, 95):.3f} ms")
cols[3].metric("P99", f"{np.percentile(arr, 99):.3f} ms")
cols[4].metric("Max", f"{np.max(arr):.3f} ms")

# Target check
target = 5.0
if np.median(arr) < target:
    st.success(f"Median latency {np.median(arr):.3f} ms < {target} ms target")
else:
    st.error(f"Median latency {np.median(arr):.3f} ms exceeds {target} ms target")

# Plots
st.subheader("Latency Distribution")
st.plotly_chart(plot_latency_histogram(latencies), use_container_width=True)

# Grid info
with st.expander("Details"):
    st.write(f"Grid: {ds.markers.grid_shape[0]}x{ds.markers.grid_shape[1]} = "
             f"{ds.markers.grid_shape[0] * ds.markers.grid_shape[1]} markers")
    st.write(f"Frames processed: {len(latencies)}")
    st.write(f"Total time: {sum(latencies):.1f} ms")
    st.write(f"Effective throughput: {len(latencies) / (sum(latencies) / 1000):.0f} fps")
