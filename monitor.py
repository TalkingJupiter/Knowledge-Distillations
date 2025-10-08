#!/usr/bin/env python3
"""
Usage:
  python monitor.py --output logs/runX/telemetry.jsonl --interval 5

This version uses `nvidia-smi` instead of NVML/pynvml and timestamps in Lubbock local time (America/Chicago).
"""

# import os, sys, time, json, signal, argparse, datetime, subprocess, shutil
# import psutil
# import zoneinfo


# # ---------------- Timezone ----------------
# CT = zoneinfo.ZoneInfo("America/Chicago")  # Central Time, auto-handles DST for Lubbock

# # ---------------- Helpers ----------------
# def _float_or_none(x):
#     if x is None:
#         return None
#     s = str(x).strip()
#     if not s or s.upper() == "N/A":
#         return None
#     try:
#         return float(s)
#     except Exception:
#         return None

# def _int_or_none(x):
#     if x is None:
#         return None
#     s = str(x).strip()
#     if not s or s.upper() == "N/A":
#         return None
#     try:
#         return int(float(s))
#     except Exception:
#         return None

# def have_nvidia_smi() -> bool:
#     return shutil.which("nvidia-smi") is not None

# # ------------- GPU telemetry via nvidia-smi -------------
# _GPU_FIELDS = [
#     "index",
#     "name",
#     "power.draw",
#     "memory.total",
#     "memory.used",
#     "utilization.gpu",
#     "utilization.memory",
#     "temperature.gpu",
#     "fan.speed",
# ]

# def _query_nvidia_smi(fields):
#     cmd = [
#         "nvidia-smi",
#         f"--query-gpu={','.join(fields)}",
#         "--format=csv,noheader,nounits",
#     ]
#     try:
#         proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
#     except FileNotFoundError:
#         return []
#     except subprocess.CalledProcessError:
#         return []
#     lines = [ln.strip() for ln in proc.stdout.strip().splitlines() if ln.strip()]
#     rows = [ [col.strip() for col in ln.split(",")] for ln in lines ]
#     return rows

# def get_gpu_info():
#     if not have_nvidia_smi():
#         return []
#     rows = _query_nvidia_smi(_GPU_FIELDS)
#     out = []
#     for r in rows:
#         idx = _int_or_none(r[0]) if len(r) > 0 else None
#         name = r[1].strip() if len(r) > 1 else None
#         power_w = _float_or_none(r[2]) if len(r) > 2 else None
#         mem_total_mib = _float_or_none(r[3]) if len(r) > 3 else None
#         mem_used_mib  = _float_or_none(r[4]) if len(r) > 4 else None
#         util_gpu = _int_or_none(r[5]) if len(r) > 5 else None
#         util_mem = _int_or_none(r[6]) if len(r) > 6 else None
#         temp_c   = _int_or_none(r[7]) if len(r) > 7 else None
#         fan_pct  = _int_or_none(r[8]) if len(r) > 8 else None

#         out.append({
#             "gpu_index": idx,
#             "gpu_name": name,
#             "power_watts": power_w,
#             "energy_mJ": None,  # `nvidia-smi` usually doesn't report cumulative energy
#             "memory_used_MB": mem_used_mib,
#             "memory_total_MB": mem_total_mib,
#             "gpu_utilization_percent": util_gpu,
#             "memory_utilization_percent": util_mem,
#             "temperature_C": temp_c,
#             "fan_speed_percent": fan_pct,
#         })
#     return out

# # ------------- CPU telemetry -------------
# def get_cpu_info():
#     cpu_util = psutil.cpu_percent(interval=None)
#     ram = psutil.virtual_memory()
#     info = {
#         "cpu_utilization_percent": cpu_util,
#         "ram_used_MB": ram.used / (1024**2),
#         "ram_total_MB": ram.total / (1024**2),
#     }
#     # Optional: CPU energy via Intel RAPL if available
#     rapl_path = "/sys/class/powercap/intel-rapl:0/energy_uj"
#     try:
#         with open(rapl_path, "r") as f:
#             energy_uj = int(f.read().strip())
#         info["cpu_energy_uj"] = energy_uj
#     except Exception:
#         info["cpu_energy_uj"] = None
#     return info

# # ------------- Main loop -------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--output", required=True, help="Output JSONL file")
#     ap.add_argument("--interval", type=int, default=5, help="Seconds between samples")
#     args = ap.parse_args()

#     if not have_nvidia_smi():
#         print("WARNING: `nvidia-smi` not found. GPU metrics will be empty.", file=sys.stderr)

#     out_dir = os.path.dirname(args.output)
#     if out_dir:
#         os.makedirs(out_dir, exist_ok=True)

#     stop = False
#     def handle_sig(*_):
#         nonlocal stop
#         stop = True
#     signal.signal(signal.SIGINT, handle_sig)
#     signal.signal(signal.SIGTERM, handle_sig)

#     with open(args.output, "a", encoding="utf-8") as f:
#         while not stop:
#             # Central Time (Lubbock) timestamp
#             ts = datetime.datetime.now(CT).isoformat()
#             entry = {
#                 "timestamp": ts,
#                 "gpus": get_gpu_info(),
#                 "cpu": get_cpu_info(),
#             }
#             f.write(json.dumps(entry) + "\n")
#             f.flush()
#             time.sleep(args.interval)

# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
"""
Usage:
  python monitor.py --output logs/runX/telemetry.jsonl --interval 5
"""

import os, sys, time, json, signal, psutil, argparse, datetime
from zoneinfo import ZoneInfo

# ------------ NVML init (GPU telemetry) ------------
try:
    import pynvml  # a.k.a. nvidia-ml-py
    pynvml.nvmlInit()
    NVML_OK = True
except Exception:
    NVML_OK = False

def _to_str(x):
    # NVML sometimes returns bytes, sometimes str
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)

def get_gpu_info():
    if not NVML_OK:
        return []
    out = []
    try:
        count = pynvml.nvmlDeviceGetCount()
    except Exception:
        return out
    for i in range(count):
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = _to_str(pynvml.nvmlDeviceGetName(h))
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            # power in Watts
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                power = None
            # total energy (mJ) if supported
            try:
                energy_mJ = pynvml.nvmlDeviceGetTotalEnergyConsumption(h) * 1000.0
            except Exception:
                energy_mJ = None
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None
            try:
                fan = pynvml.nvmlDeviceGetFanSpeed(h)
            except Exception:
                fan = None

            out.append({
                "gpu_index": i,
                "gpu_name": name,
                "power_watts": power,
                "energy_mJ": energy_mJ,
                "memory_used_MB": mem.used / (1024**2),
                "memory_total_MB": mem.total / (1024**2),
                "gpu_utilization_percent": getattr(util, "gpu", None),
                "memory_utilization_percent": getattr(util, "memory", None),
                "temperature_C": temp,
                "fan_speed_percent": fan,
            })
        except Exception:
            # keep going if one device query fails
            continue
    return out

# ------------ CPU telemetry ------------
def get_cpu_info():
    cpu_util = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    info = {
        "cpu_utilization_percent": cpu_util,
        "ram_used_MB": ram.used / (1024**2),
        "ram_total_MB": ram.total / (1024**2),
    }
    # Optional: CPU power via Intel RAPL if available
    rapl_path = "/sys/class/powercap/intel-rapl:0/energy_uj"
    try:
        with open(rapl_path, "r") as f:
            energy_uj = int(f.read().strip())
        info["cpu_energy_uj"] = energy_uj  # cumulative since boot
    except Exception:
        # Not available on all nodes / vendors / permissions
        info["cpu_energy_uj"] = None
    return info

# ------------ Main loop ------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="Output JSONL file")
    ap.add_argument("--interval", type=int, default=5, help="Seconds between samples")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    stop = False
    def handle_sig(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    lubbock_tz = ZoneInfo("America/Chicago")

    with open(args.output, "a", encoding="utf-8") as f:
        while not stop:
            ts = datetime.datetime.now(lubbock_tz).isoformat()
            entry = {
                "timestamp": ts,
                "gpus": get_gpu_info(),
                "cpu": get_cpu_info(),
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()
            time.sleep(args.interval)

if __name__ == "__main__":
    main()
