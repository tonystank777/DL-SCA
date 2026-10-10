# Viva support

**Cloud virtualization** – one physical server runs many virtual machines (VMs), each acting like its own computer.
**Multi-tenancy** – different customers (tenants) share the same physical host.
**Side-channel attack** – learning secrets from *how* a system behaves (timing, cache use), not by breaking into it.
**Why shared resources leak** – VMs share CPU cache and memory; one tenant's activity changes the timing/cache state another tenant can measure.
**Hardware performance counters (HPCs)** – CPU registers that count events such as cache references/misses and branch misses.
**Data we collect** – Research: HPC counts from a public dataset. Live: OS metrics (CPU, memory, disk, network, process count) via psutil.
**Why deep learning** – a neural network learns non-linear combinations of many counters without hand-written rules.
**How it separates normal/suspicious** – it outputs P(suspicious) from 0 to 1; above the threshold (0.5) = SUSPICIOUS.
**Dataset / source** – IEEE DataPort "Microarchitectural Attack Detection Dataset (Spectre-v1, Spectre-v2, Benign)", Kotha & Gaudiot, UC Irvine, 2026.
**Architecture** – Dense64-ReLU-Dropout-Dense32-ReLU-Dropout-Dense1-Sigmoid.
**Live detection** – collect sample → same log/scaler as training → model → probability → threshold → dashboard alert.
**Why Streamlit** – pure Python, quick live dashboards, free hosting.
**Why Streamlit Cloud can't monitor my laptop** – the app runs on Streamlit's server, not on my PC, so it can only see its own container. Hence Demo mode online, Local mode on my PC.
**Limitations** – dataset is bare-metal Spectre, not cloud VMs; live model detects CPU/memory stress, not real side channels; VM1-3 are labels; small local dataset.
**Future** – Linux VMs with `perf` HPCs, per-VM agents, LSTM on time windows.

## Thanglish explanation for your guide
"Cloud-la multiple VMs same physical hardware resources share pannum. Suspicious activity nadakkumbodhu CPU, cache maadhiri resources-oda performance pattern maaralam. Naan oru real public research dataset (hardware performance counters, Spectre attack vs normal) vachu Deep Learning model train panni, test panni, actual accuracy-ah report panren. Adhoda, en laptop-la real-time OS metrics collect panni, safe-aana CPU/memory stress workload-ah 'suspicious' nu label panni oru live model-um train panniruken. New data vandha model suspicious probability kudukkum; threshold-ku mela irundha Streamlit dashboard-la alert varum. Research dataset-la HPC features, live-la OS features — rendum different, adhunala live part ore real side-channel attack detect panradhu illa, pipeline-ah demonstrate panradhu dhaan. Streamlit Cloud-la en laptop-ah monitor panna mudiyadhu, so online-la Demo mode, local-la real monitoring."
