#!/usr/bin/env python3
"""
Monitor GPU (NVML) and CPU telemetry with Lubbock (America/Chicago) timestamps.

Design goals (research artifact friendly):
- Clear runtime checks with informative warnings (no silent failures).
- Portable: works on GPU or CPU-only machines.
- Self-documenting and reproducible.

Usage:
  python monitor.py --output logs/runX/telemetry.jsonl --interval 5
"""

import os
import time
import json
import signal
import argparse
import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Mapping

import psutil

# ======== GPU backends: nvidia-ml-py (preferred) and raw NVML (fallback) ========
NSMI_OK = False
PNVML_OK = False
nsmi: Any = None  # runtime-initialized interface to nvidia-smi Python wrapper

# Try high-level nvidia-ml-py (pynvml.smi.nvidia_smi)
try:
    from pynvml.smi import nvidia_smi  # pyright: ignore[reportMissingImports]
    nsmi = nvidia_smi.getInstance()
    if nsmi is None:
        raise RuntimeError("nvidia_smi.getInstance() returned None")
    NSMI_OK = True
except Exception as e:
    NSMI_OK = False
    nsmi = None
    print(f"[WARN] NVIDIA SMI (nvidia-ml-py) interface unavailable: {e}", flush=True)

# Try low-level NVML (pynvml) as fallback
try:
    import pynvml
    pynvml.nvmlInit()
    PNVML_OK = True
except Exception as e:
    PNVML_OK = False
    print(f"[WARN] Low-level NVML (pynvml) interface unavailable: {e}", flush=True)


# ======== Helpers ========
def _parse_number(s: Any) -> Optional[float]:
    """
    Parse values like '42 W', '1234 MiB', '17 %', '65 C' to float.
    Return None if not parseable.
    """
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    if not isinstance(s, str):
        try:
            return float(s)
        except Exception:
            return None
    txt = s.strip()
    for suf in ("W", "mW", "J", "mJ", "MiB", "GiB", "C", "%"):
        if txt.endswith(suf):
            txt = txt[: -len(suf)].strip()
    try:
        return float(txt)
    except Exception:
        return None


def _mb_from_fb_mem(mem_dict: Mapping[str, Any] | None) -> tuple[Optional[float], Optional[float]]:
    """Return (used_MB, total_MB) from NVSMI fb_memory_usage dict."""
    if not isinstance(mem_dict, Mapping):
        return None, None
    used = _parse_number(mem_dict.get("used"))
    total = _parse_number(mem_dict.get("total"))
    return used, total


# ======== GPU telemetry ========
def get_gpu_info() -> List[Dict[str, Any]]:
    """
    Returns a list of GPU telemetry dicts. Prefers nvidia-ml-py (single bulk query),
    falls back to raw NVML. If neither is available, returns [].
    """
    out: List[Dict[str, Any]] = []

    # Preferred: high-level nvidia-ml-py
    if NSMI_OK and nsmi is not None:
        try:
            q = nsmi.DeviceQuery(
                "index, name, fan.speed, temperature.gpu, power.draw, "
                "utilization.gpu, utilization.memory, fb_memory_usage"
            )
            gpus = q.get("gpu", [])
            if isinstance(gpus, dict):
                gpus = [gpus]

            for g in gpus:
                idx = g.get("index")
                name = g.get("product_name") or g.get("name")

                fan = _parse_number(g.get("fan_speed") or g.get("fan.speed"))

                temp = None
                if isinstance(g.get("temperature"), dict):
                    temp = _parse_number(g["temperature"].get("gpu_temp"))
                elif "temperature.gpu" in g:
                    temp = _parse_number(g["temperature.gpu"])

                power = None
                p = g.get("power_readings") or {}
                if isinstance(p, dict):
                    power = _parse_number(p.get("power_draw"))
                elif "power.draw" in g:
                    power = _parse_number(g["power.draw"])

                util_gpu = None
                util_mem = None
                u = g.get("utilization") or {}
                if isinstance(u, dict):
                    util_gpu = _parse_number(u.get("gpu_util"))
                    util_mem = _parse_number(u.get("memory_util"))
                else:
                    util_gpu = _parse_number(g.get("utilization.gpu"))
                    util_mem = _parse_number(g.get("utilization.memory"))

                used_mb, total_mb = _mb_from_fb_mem(g.get("fb_memory_usage"))

                rec: Dict[str, Any] = {
                    "gpu_index": idx,
                    "gpu_name": name,
                    "power_watts": power,
                    "energy_mJ": None,  # may be enriched by raw NVML below
                    "memory_used_MB": used_mb,
                    "memory_total_MB": total_mb,
                    "gpu_utilization_percent": util_gpu,
                    "memory_utilization_percent": util_mem,
                    "temperature_C": temp,
                    "fan_speed_percent": fan,
                }
                out.append(rec)
        except Exception as e:
            print(f"[WARN] nvidia-ml-py DeviceQuery failed, trying raw NVML: {e}", flush=True)
            out = []

    # Fallback: raw NVML
    if (not out) and PNVML_OK:
        try:
            count = pynvml.nvmlDeviceGetCount()
        except Exception:
            count = 0
        for i in range(count):
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name_raw = pynvml.nvmlDeviceGetName(h)
                name = name_raw.decode() if isinstance(name_raw, (bytes, bytearray)) else str(name_raw)

                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)

                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                except Exception:
                    power = None
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

                mem_used = float(getattr(mem, "used", 0)) / (1024**2)
                mem_total = float(getattr(mem, "total", 0)) / (1024**2)

                out.append(
                    {
                        "gpu_index": i,
                        "gpu_name": name,
                        "power_watts": power,
                        "energy_mJ": energy_mJ,
                        "memory_used_MB": mem_used,
                        "memory_total_MB": mem_total,
                        "gpu_utilization_percent": getattr(util, "gpu", None),
                        "memory_utilization_percent": getattr(util, "memory", None),
                        "temperature_C": temp,
                        "fan_speed_percent": fan,
                    }
                )
            except Exception:
                continue

    # Enrich energy via raw NVML if we started with nvidia-ml-py and NVML is present
    if out and NSMI_OK and PNVML_OK:
        try:
            count = pynvml.nvmlDeviceGetCount()
            for rec in out:
                idx = int(rec.get("gpu_index", -1))
                if 0 <= idx < count:
                    try:
                        h = pynvml.nvmlDeviceGetHandleByIndex(idx)
                        rec["energy_mJ"] = pynvml.nvmlDeviceGetTotalEnergyConsumption(h) * 1000.0
                    except Exception:
                        pass
        except Exception:
            pass

    return out


# ======== CPU telemetry ========
def get_cpu_info() -> Dict[str, Any]:
    #TODO: CPU utilization, frequency, temperature, load, RAM, and RAPL energy (if available).
    
    cpu_util_overall = psutil.cpu_percent(interval=None)
    cpu_util_per_core = psutil.cpu_percent(interval=None, percpu=True)
    freq = psutil.cpu_freq()
    try:
        load1, load5, load15 = os.getloadavg()
    except Exception:
        load1 = load5 = load15 = None
    ram = psutil.virtual_memory()

    info: Dict[str, Any] = {
        "cpu_utilization_percent": cpu_util_overall,
        "cpu_utilization_per_core": cpu_util_per_core,
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_frequency_MHz": (freq.current if freq else None),
        "cpu_frequency_min_MHz": (freq.min if freq else None),
        "cpu_frequency_max_MHz": (freq.max if freq else None),
        "loadavg_1min": load1,
        "loadavg_5min": load5,
        "loadavg_15min": load15,
        "ram_used_MB": float(ram.used) / (1024**2),
        "ram_total_MB": float(ram.total) / (1024**2),
    }

    # Optional: temperature sensors (platform dependent)
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            info["cpu_temperatures_C"] = {k: (v[0].current if v else None) for k, v in temps.items()}
    except Exception:
        info["cpu_temperatures_C"] = None

    # Optional: CPU energy via Intel RAPL
    rapl_path = "/sys/class/powercap/intel-rapl:0/energy_uj"
    try:
        with open(rapl_path, "r") as f:
            energy_uj = int(f.read().strip())
        info["cpu_energy_uj"] = energy_uj
    except Exception:
        info["cpu_energy_uj"] = None

    return info


# ======== Main loop ========
def main():
    ap = argparse.ArgumentParser(description="Monitor GPU/CPU telemetry in Lubbock (America/Chicago).")
    ap.add_argument("--output", required=True, help="Output JSONL file path.")
    ap.add_argument("--interval", type=int, default=5, help="Sampling interval in seconds.")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    stop = False

    def handle_sig(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    lubbock_tz = ZoneInfo("America/Chicago")
    print(f"[INFO] Monitoring started (America/Chicago) at {datetime.datetime.now(lubbock_tz).isoformat()}")

    if not NSMI_OK and not PNVML_OK:
        print("[WARN] No NVIDIA telemetry backends available; GPU metrics will be empty.", flush=True)

    with open(args.output, "a", encoding="utf-8") as f:
        while not stop:
            ts = datetime.datetime.now(lubbock_tz).isoformat()
            entry = {
                "timestamp": ts,
                "cpu": get_cpu_info(),
                "gpus": get_gpu_info(),
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()
            time.sleep(args.interval)

    print(f"[INFO] Monitoring stopped at {datetime.datetime.now(lubbock_tz).isoformat()}")


if __name__ == "__main__":
    main()
