#!/usr/bin/env bash
# Hybrid MoE baseline sweep (-mh): system info, then eval/hybrid_perf.py under a few placements,
# each writing prefill/decode speeds and dispatcher telemetry to JSON. Send back the whole output
# directory. Usage:
#   eval/hybrid_baseline.sh /path/to/exl3_model [out_dir]
# Tunables (env): CACHE_LIMIT_GB (expert cache cap, default 45), MAX_LENGTH (default 16384),
# CACHE_SIZE (default 32768), GPU (CUDA device index, default 0), RUNS (default "replicate shard")
set -euo pipefail

MODEL="${1:?usage: $0 /path/to/exl3_model [out_dir]}"
OUT="${2:-hybrid_baseline_$(date +%Y%m%d_%H%M%S)}"
CACHE_LIMIT_GB="${CACHE_LIMIT_GB:-45}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
CACHE_SIZE="${CACHE_SIZE:-32768}"
GPU="${GPU:-0}"
RUNS="${RUNS:-replicate shard}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT"

# System description: CPU flags (AMX), NUMA layout incl. SNC, memory speed, GPU and its NUMA node
{
    echo "== date";      date -Is
    echo "== git";       git -C "$HERE/.." log -1 --oneline || true
    echo "== lscpu";     lscpu
    echo "== amx";       grep -o -w 'amx_[a-z0-9]*' /proc/cpuinfo | sort -u || true
    echo "== numactl";   numactl -H || true
    echo "== hugepages"; grep -i huge /proc/meminfo || true
    echo "== thp";       cat /sys/kernel/mm/transparent_hugepage/enabled || true
    echo "== dimm";      (dmidecode -t memory 2>/dev/null | grep -E 'Configured Memory Speed|^\s+Speed' | sort | uniq -c) || echo "dmidecode needs root"
    echo "== nvidia-smi"; nvidia-smi || true
    echo "== gpu topo";  nvidia-smi topo -m || true
    echo "== gpu pcie";  nvidia-smi -i "$GPU" --query-gpu=name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,memory.used,memory.total --format=csv || true
} > "$OUT/system.txt" 2>&1

export CUDA_VISIBLE_DEVICES="$GPU"
export EXL3_HYBRID_MOE_PROFILE=1
export EXL3_NUMA_ENABLE=1
export EXL3_EXPERT_CACHE_LIMIT_GB="$CACHE_LIMIT_GB"

for run in $RUNS; do
    echo " -- run: $run"
    EXL3_NUMA_MEMORY_POLICY="$run" python "$HERE/hybrid_perf.py" \
        -m "$MODEL" -mh -cs "$CACHE_SIZE" -max_length "$MAX_LENGTH" -dr 3 \
        -json "$OUT/$run.json" 2>&1 | tee "$OUT/$run.log"
done

echo " -- done: $OUT"
