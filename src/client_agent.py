"""
DL-SCA client agent.

Run this ON THE MACHINE YOU WANT MONITORED (your laptop/desktop/VM) -- it is the
piece that makes "client-side" monitoring real when the dashboard itself is
hosted remotely (e.g. on Render). It does three things, all on YOUR machine:

  1. Reads real OS metrics every `--interval` seconds with the existing
     LocalCollector (psutil) -- exactly the same code path as Local Monitoring
     in app.py.
  2. POSTs each reading to the server's /api/ingest endpoint.
  3. Polls /api/command/<client-id> so that pressing "Start workload" on the
     web dashboard tells THIS process to run the existing SAFE SIMULATED
     SUSPICIOUS WORKLOAD (workload_generator) locally -- the server never runs
     the workload itself, it only relays the button press.

Nothing is sent anywhere except the numeric metrics listed in FEATURES
(data_collector.FEATURES) plus a timestamp and your chosen client id.

Usage:
    python -m src.client_agent --server-url https://your-app.onrender.com --client-id my-laptop
    python -m src.client_agent --server-url http://localhost:8000 --token mysecret

Stop with Ctrl+C (also stops any running workload cleanly).
"""
import argparse
import platform
import socket
import sys
import time

import requests

from src.data_collector import LocalCollector
from src.workload_generator import start_workload, stop_workload


def run(server_url: str, client_id: str, token: str, interval: float):
    server_url = server_url.rstrip("/")
    headers = {"X-Agent-Token": token} if token else {}
    collector = LocalCollector()
    workload_proc = None
    workload_state = None  # mirrors what we last acted on, so we don't re-trigger every poll

    print(f"[client_agent] machine   : {platform.node()} ({platform.system()} {platform.release()})")
    print(f"[client_agent] client_id : {client_id}")
    print(f"[client_agent] server    : {server_url}")
    print(f"[client_agent] posting every {interval}s. Ctrl+C to stop.\n")

    try:
        while True:
            # 1) read + send this machine's real metrics
            row = collector.read(vm_id=client_id, tenant="")[0]
            payload = {"client_id": client_id, **{k: row[k] for k in (
                "timestamp", "cpu_percent", "mem_percent", "disk_read_mb_s", "disk_write_mb_s",
                "net_sent_kb_s", "net_recv_kb_s", "process_count", "cpu_freq_mhz")}}
            try:
                r = requests.post(f"{server_url}/api/ingest", json=payload, headers=headers, timeout=5)
                if r.status_code == 401:
                    print("[client_agent] server rejected the request: wrong/missing --token.")
                elif not r.ok:
                    print(f"[client_agent] ingest failed: {r.status_code} {r.text[:200]}")
            except requests.RequestException as e:
                print(f"[client_agent] could not reach server: {e}")

            # 2) check whether the dashboard asked us to start/stop the SAFE workload
            try:
                r = requests.get(f"{server_url}/api/command/{client_id}", headers=headers, timeout=5)
                if r.ok:
                    cmd = r.json().get("workload")
                    if cmd == "start" and workload_state != "running":
                        duration = r.json().get("duration", 60)
                        print(f"[client_agent] SAFE SIMULATED SUSPICIOUS WORKLOAD starting for {duration}s "
                              f"(CPU+memory burn on THIS machine only; not a real attack)")
                        workload_proc = start_workload(duration)
                        workload_state = "running"
                        requests.post(f"{server_url}/api/command/{client_id}/ack", headers=headers, timeout=5)
                    elif cmd == "stop" and workload_state == "running":
                        print("[client_agent] stopping workload")
                        stop_workload(workload_proc)
                        workload_proc, workload_state = None, None
                        requests.post(f"{server_url}/api/command/{client_id}/ack", headers=headers, timeout=5)
            except requests.RequestException:
                pass  # non-fatal; just try again next loop

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[client_agent] stopping…")
        stop_workload(workload_proc)
        sys.exit(0)


def main():
    ap = argparse.ArgumentParser(description="DL-SCA client-side monitoring agent")
    ap.add_argument("--server-url", required=True, help="e.g. https://your-app.onrender.com")
    ap.add_argument("--client-id", default=socket.gethostname(), help="label shown on the dashboard (default: hostname)")
    ap.add_argument("--token", default="", help="value of AGENT_TOKEN set on the server, if any")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between readings (default 2)")
    a = ap.parse_args()
    run(a.server_url, a.client_id, a.token, a.interval)


if __name__ == "__main__":
    main()
