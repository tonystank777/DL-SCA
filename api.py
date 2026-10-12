"""
DL-SCA Render-ready server.

WHY THIS EXISTS
----------------
The original app.py is a Streamlit app. When Streamlit (or any web app) runs on a
host like Render, its Python process can only see the CPU/memory/disk/network of
the Render container it runs in -- it has NO way to read the browser visitor's own
computer. "Local Monitoring" in app.py only ever worked because you ran Streamlit
ON your own PC.

To genuinely run the SAFE SIMULATED SUSPICIOUS WORKLOAD and monitor it on the
CLIENT side while the dashboard is hosted on Render, you need two separate pieces:

  1. src/client_agent.py   -- runs on the VISITOR's own machine (python src/client_agent.py).
                               Uses the existing LocalCollector + workload_generator to read
                               REAL psutil metrics of that machine and POSTs them here.
  2. server/api.py (this)  -- deployed on Render. Receives those readings, runs them
                               through the trained live model (src/predictor.Detector),
                               stores recent history in memory, and serves a small
                               browser dashboard that polls this API.

Nothing here invents client data: every row shown as "source": "CLIENT (real OS metrics)"
came from psutil running on that specific machine, the same honest labelling style as
the rest of this project.

Run locally:   uvicorn server.api:app --reload --port 8000
Run on Render: uvicorn server.api:app --host 0.0.0.0 --port $PORT
"""
import csv
import io
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import MODELS
from src.data_collector import FEATURES

APP_DIR = Path(__file__).resolve().parent
AGENT_TOKEN = os.environ.get("AGENT_TOKEN", "")  # set this in Render's env vars; empty = no auth (demo only)
MAX_ROWS_PER_CLIENT = 2000
STALE_AFTER_S = 30  # a client with no reading in this long is shown as "offline"

app = FastAPI(title="DL-SCA Client Monitor")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

# ---- in-memory state -------------------------------------------------------
# NOTE: this is intentionally simple (a dict in process memory), matching the
# scope of the rest of this student project. On Render's free tier the process
# can restart/sleep, which clears history -- that's fine for a live demo, but
# don't rely on this for anything that must persist.
history: dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_ROWS_PER_CLIENT))
last_seen: dict[str, float] = {}
commands: dict[str, dict] = {}  # client_id -> {"workload": "start"/"stop"/None, "duration": int}

_detector = None
_detector_error = None


def get_detector():
    """Lazy-load the Keras live model so the server still boots if it hasn't been trained yet."""
    global _detector, _detector_error
    if _detector is not None or _detector_error is not None:
        return _detector
    try:
        if not (MODELS / "live_model.keras").exists():
            raise FileNotFoundError(
                "models/live_model.keras not found. Train it first: "
                "python -m src.data_collector --seconds 120 && python -m src.train_model --mode live"
            )
        from src.predictor import Detector
        _detector = Detector("live")
    except Exception as e:  # noqa: BLE001 - we want to surface any load error to the dashboard, not crash the server
        _detector_error = str(e)
    return _detector


# ---- request/response models ----------------------------------------------
class Reading(BaseModel):
    client_id: str
    timestamp: str
    cpu_percent: float
    mem_percent: float
    disk_read_mb_s: float = 0.0
    disk_write_mb_s: float = 0.0
    net_sent_kb_s: float = 0.0
    net_recv_kb_s: float = 0.0
    process_count: int = 0
    cpu_freq_mhz: Optional[float] = None


class CommandIn(BaseModel):
    workload: str  # "start" or "stop"
    duration: int = 60


def _check_token(x_agent_token: Optional[str]):
    if AGENT_TOKEN and x_agent_token != AGENT_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Agent-Token header.")


# ---- agent-facing endpoints -------------------------------------------------
@app.post("/api/ingest")
def ingest(reading: Reading, x_agent_token: Optional[str] = Header(None)):
    """Called by src/client_agent.py (running on the visitor's own machine) every ~2s."""
    _check_token(x_agent_token)
    row = reading.dict()
    cid = row.pop("client_id")
    row["source"] = "CLIENT (real OS metrics, via agent)"

    det = get_detector()
    if det is not None:
        try:
            X = [[float(row[f]) if row.get(f) is not None else 0.0 for f in FEATURES]]
            from src.preprocessing import transform_features
            Xs = transform_features(X, det.scaler, det.meta["use_log"])
            p = float(det.model.predict(Xs, verbose=0).ravel()[0])
            row["probability"] = p
            row["prediction"] = "SUSPICIOUS" if p >= det.threshold else "NORMAL"
        except Exception as e:  # noqa: BLE001
            row["probability"], row["prediction"] = None, f"error: {e}"
    else:
        row["probability"], row["prediction"] = None, "model not loaded"

    history[cid].append(row)
    last_seen[cid] = time.time()
    return {"ok": True, "stored_rows": len(history[cid])}


@app.get("/api/command/{client_id}")
def get_command(client_id: str, x_agent_token: Optional[str] = Header(None)):
    """Polled by the agent; tells it whether the dashboard asked for a workload start/stop."""
    _check_token(x_agent_token)
    cmd = commands.get(client_id)
    return cmd or {"workload": None}


@app.post("/api/command/{client_id}/ack")
def ack_command(client_id: str, x_agent_token: Optional[str] = Header(None)):
    """Agent calls this once it has acted on a command, so it isn't re-triggered every poll."""
    _check_token(x_agent_token)
    commands[client_id] = {"workload": None}
    return {"ok": True}


# ---- dashboard-facing endpoints --------------------------------------------
@app.get("/api/clients")
def list_clients():
    now = time.time()
    return [
        {"client_id": cid, "online": (now - last_seen.get(cid, 0)) < STALE_AFTER_S, "rows": len(history[cid])}
        for cid in history
    ]


@app.get("/api/latest")
def latest(client_id: str, n: int = 200):
    rows = list(history.get(client_id, []))[-n:]
    return {
        "rows": rows,
        "detector_loaded": get_detector() is not None,
        "detector_error": _detector_error,
        "online": (time.time() - last_seen.get(client_id, 0)) < STALE_AFTER_S if client_id in last_seen else False,
    }


@app.post("/api/command/{client_id}")
def set_command(client_id: str, cmd: CommandIn):
    """Dashboard calls this when you press Start/Stop workload. The AGENT on that
    client's machine picks it up on its next poll and runs the SAFE SIMULATED
    SUSPICIOUS WORKLOAD locally -- the server never runs it itself."""
    if cmd.workload not in ("start", "stop"):
        raise HTTPException(400, "workload must be 'start' or 'stop'")
    commands[client_id] = {"workload": cmd.workload, "duration": cmd.duration}
    return {"ok": True}


@app.get("/api/export/{client_id}")
def export_csv(client_id: str):
    rows = list(history.get(client_id, []))
    if not rows:
        raise HTTPException(404, "No data for this client yet.")
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={client_id}_history.csv"},
    )


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/")
def dashboard():
    return FileResponse(str(APP_DIR / "static" / "dashboard.html"))
