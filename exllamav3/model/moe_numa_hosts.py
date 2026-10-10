"""
Per-NUMA-node CPU MoE workers (hybrid MoE and EXL3_NUMA_ENABLE).

A dual-socket host is two execution and memory domains, not one pool of cores: a worker thread
on socket 1 that streams expert weights out of socket 0's DRAM pays the inter-socket link on every
byte, and one pool spanning both sockets synchronizes across it at every phase barrier. Following
the per-node sub-pool organisation of KTransformers' CPUInfer/WorkerPool (Apache-2.0; mechanism
only, no code), each NUMA node gets its own existing CPU MoE worker process (MoeCpuHost: spawned
child, spin-parked thread pool, pinned job ring), confined to the node's CPUs, with its expert
arena bound to the node's memory. Nothing in the worker itself changes: the pool derives its core
order from the process affinity mask, and first-touch plus mbind keep its pages local.

Two ways to use the group:
- hybrid MoE (moe_hybrid_dispatch): experts of one layer are sharded across the nodes (or
  replicated, EXL3_NUMA_MEMORY_POLICY=replicate) and both nodes work on the same layer at once;
- classic -mcl / -mcs with EXL3_NUMA_ENABLE=1: whole layers are assigned to nodes round-robin, so
  each layer's experts live and run on one node (no expert-level dispatch needed).

Topology comes from util/numa_topology (sysfs, cpuset, EXL3_NUMA_CPU_MAP override). A GPU whose
node cannot be determined stays "unknown": GPU-adjacent buffers then get default placement.
"""

from __future__ import annotations
import os
import sys
from ..util.numa_topology import get_topology, gpu_numa_node, numa_debug, format_cpulist


def numa_requested() -> bool:
    return os.environ.get("EXL3_NUMA_ENABLE", "0") not in ("", "0")


def memory_policy() -> str:
    v = os.environ.get("EXL3_NUMA_MEMORY_POLICY", "shard").strip().lower()
    if v not in ("shard", "replicate"):
        raise ValueError("EXL3_NUMA_MEMORY_POLICY must be shard or replicate")
    return v


def shard_layout() -> str:
    v = os.environ.get("EXL3_NUMA_SHARD", "mod").strip().lower()
    if v not in ("mod", "blocked"):
        raise ValueError("EXL3_NUMA_SHARD must be mod or blocked")
    return v


def pin_threads_enabled() -> bool:
    """Same rule as the worker pool's pin_threads_enabled() (cpu/moe_mul1.cpp): Linux pins only
    with EXL3_MOE_CPU_PIN=1, other platforms unless EXL3_MOE_CPU_PIN=0"""
    v = os.environ.get("EXL3_MOE_CPU_PIN", "")
    if sys.platform.startswith("linux"):
        return v[:1] == "1"
    return v[:1] != "0"


def _memory_node(topo, n) -> int | None:
    """Node to bind node n's worker memory to: n itself, or for a memoryless CPU node (MemTotal 0)
    the nearest node with memory by the distance row; None (default policy) when unknown"""
    if n.mem_total is None or n.mem_total > 0:
        return n.id
    order = sorted(m.id for m in topo.nodes)
    cands = [m for m in topo.nodes if m.mem_total]
    if not cands:
        return None
    if n.distances and len(n.distances) == len(order):
        return min(cands, key = lambda m: (n.distances[order.index(m.id)], m.id)).id
    return None


def plan_workers(topo, gpu_node: int | None, host_cores: int, threads: int | None, numa: bool,
                 max_nodes: int | None = None) -> dict:
    """Worker placement: one entry per CPU node (or one unconfined entry when NUMA is off or the
    host has a single node). The host process keeps `host_cores` physical cores (both SMT
    siblings) on the GPU's node when known, else on the lowest CPU node (a placement choice, not
    a locality claim). Returns dict(workers = [dict(node, cpus, threads, mem_node)], host_cpus)"""
    nodes = topo.cpu_nodes() if numa else []
    if len(nodes) <= 1:
        return dict(workers = [dict(node = None, cpus = None, threads = threads, mem_node = None)],
                    host_cpus = None, numa = False)
    if max_nodes is not None and len(nodes) > max_nodes:
        raise RuntimeError(
            f"NUMA: {len(nodes)} CPU nodes are available but the hybrid MoE dispatcher supports at most "
            f"{max_nodes} worker nodes. Restrict the CPUs (EXL3_NUMA_CPU_MAP, taskset/numactl, cpuset) to "
            f"{max_nodes} nodes or fewer, or set EXL3_NUMA_ENABLE=0 for one worker")
    res_node = gpu_node if gpu_node is not None and topo.node(gpu_node) in nodes else nodes[0].id
    workers, host_cpus = [], []
    for n in nodes:
        cores = list(n.cores)
        if n.id == res_node and host_cores > 0 and len(cores) > host_cores:
            for c in cores[-host_cores:]:
                host_cpus += c
            cores = cores[:-host_cores]
        cpus = sorted(cpu for c in cores for cpu in c)
        workers.append(dict(node = n.id, cpus = cpus, threads = threads or max(1, len(cores)),
                            mem_node = _memory_node(topo, n)))
    return dict(workers = workers, host_cpus = sorted(host_cpus) or None, numa = True)


class HostGroup:
    """The CPU MoE workers of one model component, one per NUMA node (one total without NUMA).
    Hosts are registered in config.moe_cpu_hosts under "<comp>@<i>", so the loader's
    commit_module / ensure_started / begin_pass hooks reach every one of them"""

    def __init__(self, config, comp: str, device_index: int | None, home: str | None, telemetry: bool,
                 max_nodes: int | None = None):
        from .moe_cpu_host import MoeCpuHost, TUNING
        self.config = config
        self.comp = comp
        numa = numa_requested()
        self.topo = get_topology() if numa else None
        self.gpu_node = gpu_numa_node(device_index) if (numa and device_index is not None) else None
        ip = config.infer_params
        cfg_threads = getattr(ip, "moe_cpu_threads" if comp == "text" else "draft_moe_cpu_threads", None)
        env_threads = os.environ.get("EXL3_MOE_CPU_THREADS")
        threads = cfg_threads or (int(env_threads) if env_threads else None)
        self.plan = plan_workers(self.topo, self.gpu_node, TUNING.host_cores, threads, numa, max_nodes) if numa else \
            plan_workers(None, None, 0, threads, False)
        self.numa = self.plan["numa"]
        if numa and not self.numa:
            print(" -- NUMA: single CPU node (or topology unavailable), one CPU MoE worker")
        if numa and self.numa and self.gpu_node is None:
            print(" -- NUMA: GPU node unknown (set EXL3_NUMA_GPU_NODE=dev:node to declare it); "
                  "GPU-adjacent host buffers keep default placement")
        hosts = getattr(config, "moe_cpu_hosts", None)
        if hosts is None:
            hosts = {}
            config.moe_cpu_hosts = hosts
        self.hosts = []
        for i, w in enumerate(self.plan["workers"]):
            spec = dict(index = i, node = w["node"], cpus = w["cpus"], mem_node = w["mem_node"],
                        threads = w["threads"], home = home == "native_copy",
                        home_node = self.gpu_node, map_arena = home == "worker_arena",
                        telemetry = telemetry, handoff_node = self.gpu_node)
            h = MoeCpuHost(config, numa = spec)
            hosts[f"{comp}@{i}"] = h
            self.hosts.append(h)
        # Host index whose node holds the GPU (preferred node for replicated experts), -1 unknown
        self.pref = next((i for i, w in enumerate(self.plan["workers"])
                          if w["node"] is not None and w["node"] == self.gpu_node), -1)
        self._host_pinned = False
        if numa_debug() and self.topo is not None:
            print(self.topo.describe())
            for i, w in enumerate(self.plan["workers"]):
                print(f" -- NUMA worker {i}: node {w['node']}, cpus {format_cpulist(w['cpus'] or [])}, "
                      f"{w['threads']} threads")
            if self.plan["host_cpus"]:
                print(f" -- NUMA host cpus: {format_cpulist(self.plan['host_cpus'])}")

    @property
    def num_nodes(self) -> int:
        return len(self.hosts)

    def shard(self, num_experts: int) -> list[list[int]]:
        """Experts held by each host: disjoint shards (default) or every expert everywhere"""
        n = len(self.hosts)
        if n == 1 or memory_policy() == "replicate":
            return [list(range(num_experts)) for _ in range(n)]
        if shard_layout() == "blocked":
            # Contiguous, balanced: sizes differ by at most one, none empty while n <= E
            base, extra = divmod(num_experts, n)
            out, a = [], 0
            for i in range(n):
                b = a + base + (1 if i < extra else 0)
                out.append(list(range(a, b)))
                a = b
            return out
        return [list(range(i, num_experts, n)) for i in range(n)]

    def after_start(self):
        """Confine the host process to its reserved cores when the pools pin their threads (the
        pool's own rule, pin_threads_enabled), mirroring the single-worker placement. A plan with
        one unconfined worker is placed by MoeCpuHost itself (classic placement)"""
        from .moe_cpu_affinity import apply_process_affinity
        if self._host_pinned or not self.numa or not self.plan["host_cpus"]:
            return
        if not any(h.started for h in self.hosts):
            return
        if not pin_threads_enabled():
            return
        self._host_pinned = True
        err = apply_process_affinity(self.plan["host_cpus"])
        if err is not None:
            print(f" !! NUMA host affinity: {err}; host threads left unpinned")


def legacy_host_for_layer(config, comp: str, device, key: str) -> "object":
    """-mcl / -mcs with EXL3_NUMA_ENABLE=1: the worker host for the next offloaded layer of this
    component (whole layers round-robin over the nodes). None when NUMA is off, so the caller
    keeps the classic single host"""
    if not numa_requested():
        return None
    groups = getattr(config, "moe_numa_groups", None)
    if groups is None:
        groups = {}
        config.moe_numa_groups = groups
    g = groups.get(comp)
    if g is None:
        import torch
        idx = torch.device(device).index if device is not None else None
        g = HostGroup(config, comp, idx, home = None, telemetry = False)
        g.next = 0
        g.by_key = {}
        groups[comp] = g
    # An autosplit rollback retry re-registers the same layer: keep it on its first host
    h = g.by_key.get(key)
    if h is None:
        h = g.hosts[g.next % len(g.hosts)]
        g.next += 1
        g.by_key[key] = h
    return h
