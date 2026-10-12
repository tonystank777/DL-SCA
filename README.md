# DL-SCA: Deep Learning-Based Side-Channel Attack Detection in a Multi-Tenant Cloud Virtualization Environment

> Defensive college prototype. It does **not** implement any real attack.

## 1. Abstract
Virtual machines of different tenants share CPU, cache and memory of one physical host. Abnormal resource behaviour can hint at co-resident interference. DL-SCA trains a Keras neural network to classify behaviour as NORMAL or SUSPICIOUS and shows results in a Streamlit dashboard.

## 2. Problem statement
Detect suspicious performance behaviour in shared (multi-tenant) virtualised infrastructure.

## 3. Motivation
Side-channel attacks leak information through shared hardware and are invisible to normal access control. Performance data is one defensive signal (research uses hardware performance counters, HPCs).

## 4. Objectives
Collect performance data → preprocess → train DL model → evaluate honestly → live NORMAL/SUSPICIOUS prediction → dashboard with alerts and history.

## 5. Proposed solution — TWO clearly separate models
| | Research model | Live model |
|---|---|---|
| Data | REAL public HPC dataset | Metrics you record on your PC |
| Features | Linux `perf` hardware counters (cache/branch) | psutil OS metrics (CPU, memory, disk, network, process count) |
| Label 1 means | Spectre + Flush+Reload / Prime+Probe activity | SAFE SIMULATED SUSPICIOUS WORKLOAD |
| Purpose | Training / evaluation on real research data | Real-time demo |

The feature sets are **different and are not mapped to each other**. Windows/psutil cannot read HPCs, so the live model does **not** detect side-channel attacks; it demonstrates the detection pipeline on abnormal resource patterns.

## 6. Architecture / 7. Workflow
`VM/host → psutil collector → dataset/local data → preprocessing (log1p, scaler fit on train) → Keras model → evaluation → save model+scaler → live sample → same preprocessing → P(SUSPICIOUS) → threshold → NORMAL/SUSPICIOUS → Streamlit alert + history CSV`

## 8. Dataset (verified from the original page)
- **Name:** Microarchitectural Attack Detection Dataset (Spectre-v1, Spectre-v2, and Benign Workloads)
- **Authors:** Jaya Keshava Chandra Kotha, Jean-Luc Gaudiot (UC Irvine), IEEE DataPort, 2026
- **URL:** https://ieee-dataport.org/documents/microarchitectural-attack-detection-dataset-spectre-v1-spectre-v2-and-benign-workloads — DOI 10.21227/xxx6-v468
- **CSV mirror linked from that page:** https://github.com/jayakeshav/HPC_Master_Data_Sets
- **Paper:** none listed; the page says the citation is "to be added".
- **Content:** HPC time series from 5 physical machines (Intel i7/i9, AMD A4-5000, ARM Cortex-A76 on Raspberry Pi 5), captured with Linux `perf` in 3 non-multiplexed counter batches. Attacks: Spectre-v1+Flush+Reload, Spectre-v1+Prime+Probe, Spectre-v2+Flush+Reload. Benign: idle, `stress-ng` CPU/VM, OpenSSL SHA-256, GCC, tar, dd, find.
- **Label:** derived per row from attack "probe" columns (>0 → 1), as documented by the authors.
- **Feature names / sample count:** *I could not download the files while preparing this project, so I do not state them.* `train_model.py --mode research` prints the real column list, coverage and counts — copy those into your report.
- **License:** not stated on the pages I read. Check the IEEE DataPort page before redistributing (that is why `data/raw/` is git-ignored).
- **Access:** the IEEE DataPort file download needs an IEEE DataPort login/subscription; the GitHub link on the same page is public.

### Limitations (important)
Bare-metal machines, **not VMs, not cloud, not multi-tenant**. Attack type is Spectre (transient execution) with cache side channels. Labels are inferred from probe events, so attack files also contain rows labelled 0. Counter sets differ per device/batch. Time-series rows within a run are correlated, so the split is done **by run file**.

### Other candidates checked
Search results were mostly papers/patents. A 2025 arXiv paper (2501.17123) describes its own 30,000-sample perf dataset (Flush+Reload/Prime+Probe vs AES/RSA) but I found no public download. The same authors list a second, "cleaned" HPC dataset on IEEE DataPort that I did not verify in detail. No verified public dataset of *cloud/multi-tenant VM* side-channel data was found.

## 9. Technologies
Python, TensorFlow/Keras, pandas, NumPy, scikit-learn, psutil, matplotlib, Streamlit, Git/GitHub.

## 10. Modules
`dataset_loader` (read real CSVs) · `preprocessing` (features, grouped split, scaler) · `train_model` · `evaluate_model` · `data_collector` (psutil + recorder) · `workload_generator` (safe workload) · `monitoring` (tenants, demo source) · `predictor` · `app.py`.

## 11. Model
Input → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.3) → Dense(1, sigmoid). Binary cross-entropy, Adam, batch 64, ≤60 epochs, early stopping on validation loss, class weights for imbalance. 0 = NORMAL, 1 = SUSPICIOUS, threshold 0.5 (adjustable slider).

## 12. Setup (Windows + VS Code)
Use **Python 3.11** (TensorFlow does not support the newest Python versions immediately).
```powershell
cd DL-SCA
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
If PowerShell blocks activation: `Set-ExecutionPolicy -Scope Process Bypass`. In VS Code: Ctrl+Shift+P → *Python: Select Interpreter* → `.venv`.

### Dataset setup
1. Open https://github.com/jayakeshav/HPC_Master_Data_Sets → Code → Download ZIP.
2. Extract into `data/raw/` (so `Intel_PC_1` is somewhere below it).

## 13. Run order
```powershell
python -m src.train_model --mode research          # real dataset; prints columns; saves metrics + plots
python -m src.data_collector --seconds 120         # records normal, then SAFE SIMULATED workload (~5 min)
python -m src.train_model --mode live              # trains the live model
streamlit run app.py
```
If research training says too few features/rows, read the printed coverage table and retry with `--min-coverage 0.3` or `--device Intel_PC_2`.

## 14. Testing
1. Demo mode → Start: three tenants stream; some episodes labelled "stress-like" should turn red.
2. Local mode → Start → "Start workload": CPU rises and probability should climb; Stop workload: back to green.
3. Legitimate heavy apps (games, video export) may cause **false positives** — the live model only knows CPU/memory pressure.

## 15a. Render deployment — real CLIENT-side workload test

**Why this needed a new piece:** a web host's Python process (Render, Streamlit Cloud,
anywhere) can only see *its own container's* CPU/RAM. It cannot read a website visitor's
laptop. So "Local Monitoring" in `app.py` only ever worked when you ran Streamlit
*on your own PC*. To really run the SAFE SIMULATED SUSPICIOUS WORKLOAD on the **client's**
machine while the dashboard lives on Render, this project now ships two extra pieces:

| Piece | Runs where | What it does |
|---|---|---|
| `server/api.py` (FastAPI) | **Render** | Receives metrics, runs the trained live model on them, serves the browser dashboard at `/` |
| `src/client_agent.py` | **the client's own machine** | Reads REAL psutil metrics (reusing `LocalCollector`) and the SAFE workload generator (reusing `workload_generator`), POSTs readings to Render, and starts/stops the workload locally when told to from the dashboard |

The browser dashboard never runs the workload itself — it only sends a "start/stop"
command that the agent, running on the target machine, acts on. Every number shown is
labelled `"CLIENT (real OS metrics, via agent)"`, same honesty convention as the rest
of this project.

### Deploy the server to Render
1. Push this repo to GitHub (including `render.yaml`).
2. Render dashboard → **New → Blueprint** → pick the repo → Render reads `render.yaml`
   and creates a Python web service with:
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn server.api:app --host 0.0.0.0 --port $PORT`
   - Health check: `/healthz`
   - An auto-generated `AGENT_TOKEN` env var (open it in the Render dashboard and copy
     the value — the agent needs it so random internet visitors can't post fake data).
3. For real ML predictions, `models/live_model.keras` + `live_scaler.pkl` + `live_meta.json`
   must be committed (train them first, see §13). Without them the server still runs and
   shows live CPU/memory/network charts, just with `"model not loaded"` instead of a
   prediction — it doesn't crash.
4. **Memory note:** TensorFlow is heavy; Render's free tier (512 MB) can OOM. If that
   happens, upgrade the instance type or skip the model (metrics-only mode still works).

### Run the agent on the client machine
```bash
pip install -r requirements.txt     # needs psutil + requests at minimum
python -m src.client_agent --server-url https://<your-app>.onrender.com --token <AGENT_TOKEN> --client-id my-laptop
```
Leave it running. Open `https://<your-app>.onrender.com/` in a browser, pick `my-laptop`
from the dropdown, and press **Start SAFE workload** — CPU/memory will climb on *that*
machine (not Render's server) and the probability chart will react, same as the old
Local Monitoring mode, just now visible from anywhere.

### API reference (server/api.py)
- `POST /api/ingest` — agent → server, one reading (needs `X-Agent-Token` header if `AGENT_TOKEN` is set)
- `GET /api/command/{client_id}` / `POST .../ack` — agent polls for start/stop requests
- `POST /api/command/{client_id}` — dashboard → server, `{"workload":"start","duration":60}`
- `GET /api/clients`, `GET /api/latest?client_id=...`, `GET /api/export/{client_id}` — dashboard reads

### What did NOT change
`app.py` (Streamlit) and everything in §11–§16 below is untouched — it still works
exactly as before for local use or Streamlit Cloud's Demo mode.

## 15. GitHub + Streamlit Cloud
```powershell
git init; git add .; git commit -m "DL-SCA"
git branch -M main
git remote add origin https://github.com/<you>/DL-SCA.git
git push -u origin main
```
Then share.streamlit.io → *Create app* → choose repo, branch `main`, main file `app.py` → *Advanced settings* → Python 3.11 → Deploy → open the URL.
`models/` must be committed (trained before pushing). **Streamlit Cloud cannot read your PC's CPU/VMs** — the app runs on Streamlit's server, so use **Demo mode** there. Local mode only works when you run it on your own machine (or inside each VM).

## 16. Limitations
Live and research models are separate; no HPCs on Windows; VM1/2/3 are labels unless run in real VMs; live model trained on a few minutes of one machine; demo data is synthetic; not a production security tool.

## 17. Future work
Linux VMs with `perf` HPC collection (same features as the dataset), per-VM agents sending telemetry to the dashboard, LSTM over time windows, test on other devices/attacks.

## 18. References
IEEE DataPort dataset above (Kotha & Gaudiot, 2026). Background: Yarom & Falkner, *Flush+Reload*, USENIX Security 2014; Zhang et al., *CloudRadar* (search title to read the paper).
