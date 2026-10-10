"""
Hybrid MoE baseline: eval/perf.py (prefill and decode speed over context lengths) with the hybrid
dispatchers' telemetry written to a JSON file afterwards (cache hits, lane costs, measured PCIe and
CPU costs, per-device cache state). Run with -mh; see eval/hybrid_baseline.sh for the full sweep.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import time
import torch
import perf
from exllamav3 import model_init

_loaded = {}
_init = model_init.init


def _capture_init(args, *a, **kw):
    out = _init(args, *a, **kw)
    _loaded["config"] = out[1]
    return out


def _wrap(name):
    fn = getattr(perf, name)

    def timed(*a, **kw):
        t0 = time.perf_counter()
        res = fn(*a, **kw)
        if not kw.get("warmup"):
            _loaded.setdefault("results", {})[name] = (
                {f"{k[0]}x{k[1]}" if isinstance(k, tuple) else str(k): v for k, v in res.items()}
                if isinstance(res, dict) else None
            )
            _loaded.setdefault("seconds", {})[name] = time.perf_counter() - t0
        return res
    setattr(perf, name, timed)


def dump(path, args):
    config = _loaded.get("config")
    disp = (getattr(config, "moe_hybrid_dispatchers", None) or {}) if config is not None else {}
    env = {k: v for k, v in os.environ.items() if k.startswith(("EXL3_", "CUDA_VISIBLE"))}
    out = dict(
        args = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))},
        env = env,
        results = _loaded.get("results", {}),
        seconds = _loaded.get("seconds", {}),
        hybrid = {comp: d.summary() for comp, d in disp.items()},
    )
    if torch.cuda.is_available():
        out["vram"] = {str(i): dict(zip(("free", "total"), torch.cuda.mem_get_info(i)))
                       for i in range(torch.cuda.device_count())}
    with open(path, "w") as f:
        json.dump(out, f, indent = 2, default = str)
    print(f" -- Hybrid telemetry written to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    model_init.add_args(
        parser,
        default_cache_size = 32768,
        default_autosplit_max_batch_size = 1,
        default_chunk_size = 4096,
    )
    parser.add_argument("-max_length", "--max_length", type = int, help = "Max context length to measure (default: 32768)", default = 32768)
    parser.add_argument("-spf", "--skip_prefill", action = "store_true", help = "Skip measuring prefill speed")
    parser.add_argument("-dr", "--decode_reps", type = int, default = 1, help = "Repeat each decode measurement this many times")
    parser.add_argument("-sg", "--skip_gen", action = "store_true", help = "Skip measuring generation speed")
    parser.add_argument("-swu", "--skip_warmup", action = "store_true", help = "Skip warmup passes")
    parser.add_argument("-short", "--short_prefill", action = "store_true", help = "Test short-prefill/batch throughput")
    parser.add_argument("-sd", "--spec_dec", action = "store_true", help = "Test spec-decode seqlens 1..4")
    parser.add_argument("-json", "--json_out", type = str, default = "hybrid_perf.json", help = "Telemetry output path")
    _args = parser.parse_args()
    perf.model_init.init = _capture_init
    _wrap("measure_prefill")
    _wrap("measure_generate")
    try:
        perf.main(_args)
    finally:
        dump(_args.json_out, _args)
