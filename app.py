"""DL-SCA Streamlit dashboard.  Run:  streamlit run app.py"""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.config import MODELS
from src.data_collector import LocalCollector
from src.monitoring import DemoSource, TENANTS
from src.predictor import Detector
from src.workload_generator import start_workload, stop_workload

st.set_page_config(page_title="DL-SCA", page_icon="🛡️", layout="wide")
st.title("🛡️ DL-SCA: Deep Learning-Based Side-Channel Attack Detection")

if not (MODELS / "live_model.keras").exists():
    st.error("Live model not found. Run: python -m src.data_collector --seconds 120  then  "
             "python -m src.train_model --mode live  (see README).")
    st.stop()


@st.cache_resource
def get_detector():
    return Detector("live")


det = get_detector()
ss = st.session_state
ss.setdefault("running", False)
ss.setdefault("history", [])
ss.setdefault("workload", None)
ss.setdefault("mode", None)

with st.sidebar:
    mode = st.radio("Mode", ["Demo / Simulation", "Local Monitoring"],
                    help="Streamlit Cloud can only use Demo. Local Monitoring reads THIS computer via psutil.")
    vm_id = st.selectbox("VM / Tenant shown", list(TENANTS), format_func=lambda v: f"{v} → {TENANTS[v]}")
    threshold = st.slider("Decision threshold", 0.05, 0.95, float(det.threshold), 0.05)
    c1, c2 = st.columns(2)
    if c1.button("▶ Start", use_container_width=True):
        ss.running = True
    if c2.button("■ Stop", use_container_width=True):
        ss.running = False
    if mode == "Local Monitoring":
        st.caption("SAFE SIMULATED SUSPICIOUS WORKLOAD (CPU+memory burn, 60 s; not a real attack)")
        w1, w2 = st.columns(2)
        if w1.button("Start workload", use_container_width=True) and ss.workload is None:
            ss.workload = start_workload(60)
        if w2.button("Stop workload", use_container_width=True):
            stop_workload(ss.workload); ss.workload = None
    if st.button("Clear history"):
        ss.history = []

if ss.mode != mode:                                     # switching mode resets everything
    ss.mode, ss.history = mode, []
    ss.source = LocalCollector() if mode == "Local Monitoring" else DemoSource(det.meta["stats"])
demo = mode == "Demo / Simulation"


@st.fragment(run_every=2 if ss.running else None)
def live_panel():
    if ss.running:
        rows = ss.source.read() if demo else ss.source.read(vm_id, TENANTS[vm_id])
        for r, p in zip(rows, det.predict_batch(rows)):
            r["probability"] = float(p)
            r["prediction"] = "SUSPICIOUS" if p >= threshold else "NORMAL"
        ss.history = (ss.history + rows)[-2000:]

    st.subheader("🟢 Monitoring Active" if ss.running else "🔴 Monitoring Inactive")
    if demo:
        st.warning("⚠️ DEMO MODE — synthetic values drawn from statistics of a local recording. "
                   "This is NOT real cloud telemetry and NOT a real attack.")
    if not ss.history:
        st.info("Press ▶ Start in the sidebar.")
        return
    df = pd.DataFrame(ss.history)
    df["net_kb_s"] = df["net_sent_kb_s"] + df["net_recv_kb_s"]
    df["disk_mb_s"] = df["disk_read_mb_s"] + df["disk_write_mb_s"]
    cur = df[df["vm_id"] == vm_id]
    cur = cur if not cur.empty else df
    last = cur.iloc[-1]

    a = st.columns(4)
    a[0].metric("CPU usage", f"{last.cpu_percent:.1f}%")
    a[1].metric("Memory usage", f"{last.mem_percent:.1f}%")
    a[2].metric("Network", f"{last.net_kb_s:.1f} KB/s")
    a[3].metric("Disk", f"{last.disk_mb_s:.2f} MB/s")
    b = st.columns(3)
    b[0].metric("VM / Tenant", f"{last.vm_id} / {last.tenant}")
    b[1].metric("Suspicious probability", f"{last.probability:.2f}")
    b[2].metric("Detection status", last.prediction)
    if last.prediction == "SUSPICIOUS":
        st.error(f"🔴 SUSPICIOUS BEHAVIOR DETECTED on {last.vm_id} ({last.tenant}) — "
                 f"probability {last.probability:.2f} ≥ threshold {threshold:.2f}")
    else:
        st.success(f"🟢 NORMAL BEHAVIOR — probability {last.probability:.2f}")

    plot = cur.tail(60).reset_index(drop=True)
    g = st.columns(2)
    g[0].caption("CPU usage (%)"); g[0].line_chart(plot["cpu_percent"])
    g[1].caption("Memory usage (%)"); g[1].line_chart(plot["mem_percent"])
    g = st.columns(2)
    g[0].caption("Network activity (KB/s)"); g[0].line_chart(plot["net_kb_s"])
    g[1].caption("Suspicious probability"); g[1].line_chart(plot["probability"])

    st.markdown("#### Detection history")
    cols = ["timestamp", "vm_id", "tenant", "source", "cpu_percent", "mem_percent", "net_kb_s",
            "prediction", "probability"] + (["demo_scenario"] if demo else [])
    out = df[cols].iloc[::-1]
    st.dataframe(out.head(200), use_container_width=True, hide_index=True)
    st.download_button("⬇ Download history (CSV)", out.to_csv(index=False), "detection_history.csv",
                       "text/csv", key="dl_history")


tab1, tab2, tab3 = st.tabs(["Live dashboard", "Research dataset results", "Data honesty"])
with tab1:
    live_panel()
with tab2:
    mp = MODELS / "research_metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        st.caption("Model trained/tested on the REAL IEEE DataPort HPC dataset (Spectre-v1/v2 vs benign). "
                   "Grouped-by-run test split. Values below were measured, not typed in.")
        c = st.columns(4)
        for col, k in zip(c, ["accuracy", "precision", "recall", "f1"]):
            col.metric(k.capitalize(), f"{m[k]:.4f}")
        st.write("Confusion matrix [[TN, FP], [FN, TP]]:", m["confusion_matrix"], "| test rows:", m["n_test"])
        imgs = [MODELS / f"research_{n}.png" for n in ("accuracy", "loss", "confusion_matrix", "class_distribution")]
        cols = st.columns(4)
        for col, p in zip(cols, imgs):
            if p.exists():
                col.image(str(p))
    else:
        st.info("Run: python -m src.train_model --mode research  (see README, Dataset setup)")
with tab3:
    st.markdown("""
| Item | What it really is |
|---|---|
| Research model | Trained on a REAL public dataset of hardware performance counters (bare-metal machines, Spectre attacks) |
| Live model | Trained on OS metrics I recorded locally; label 1 = **SAFE SIMULATED SUSPICIOUS WORKLOAD** (CPU+memory burn) |
| Local Monitoring | REAL OS-level metrics of this computer (psutil). No hardware performance counters. |
| Demo mode | SYNTHETIC values, clearly labelled. Not cloud telemetry. |
| VM1/VM2/VM3 | Logical labels. Real per-VM data needs this app running inside each VM. |
""")
    st.json(LocalCollector.availability())
