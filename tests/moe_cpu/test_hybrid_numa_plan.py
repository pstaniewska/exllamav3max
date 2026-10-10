"""
Per-NUMA worker planning for the hybrid MoE / EXL3_NUMA_ENABLE (model/moe_numa_hosts.py): which
CPUs each node's worker gets, where the host process keeps its reserved cores, how experts are
sharded or replicated, and that an unknown GPU node is never guessed. Topologies are built from
synthetic sysfs trees through util/numa_topology.discover. No GPU needed.
"""

import os

import pytest

from testlib import hybrid_isolated

pytestmark = pytest.mark.nogpu

H = hybrid_isolated.load()


def make_sysfs(root, nodes, mem_kb = None):
    """nodes: {node_id: [(cpu, core_id, package), ...]}; mem_kb: {node_id: MemTotal kB}"""
    base = os.path.join(root, "devices", "system")
    for nid, cpus in nodes.items():
        nd = os.path.join(base, "node", f"node{nid}")
        os.makedirs(nd, exist_ok = True)
        with open(os.path.join(nd, "cpulist"), "w") as f:
            f.write(",".join(str(c) for c, _, _ in cpus) + "\n")
        with open(os.path.join(nd, "meminfo"), "w") as f:
            tot = (mem_kb or {}).get(nid, 1048576)
            f.write(f"Node {nid} MemTotal:       {tot} kB\nNode {nid} MemFree:        {tot // 2} kB\n"
                    f"Node {nid} Inactive(file):      0 kB\n")
        with open(os.path.join(nd, "distance"), "w") as f:
            f.write(" ".join("10" if o == nid else "21" for o in nodes) + "\n")
        for c, core, pkg in cpus:
            td = os.path.join(base, "cpu", f"cpu{c}", "topology")
            os.makedirs(td, exist_ok = True)
            with open(os.path.join(td, "core_id"), "w") as f:
                f.write(f"{core}\n")
            with open(os.path.join(td, "physical_package_id"), "w") as f:
                f.write(f"{pkg}\n")


def dual_socket(tmp_path):
    # 4 cores per socket, SMT: node 0 cpus 0-3 + siblings 8-11, node 1 cpus 4-7 + 12-15
    nodes = {
        0: [(c, c, 0) for c in range(4)] + [(c + 8, c, 0) for c in range(4)],
        1: [(c, c - 4, 1) for c in range(4, 8)] + [(c + 8, c - 4, 1) for c in range(4, 8)],
    }
    make_sysfs(str(tmp_path), nodes)
    return H.topo.discover(sysfs_root = str(tmp_path), cgroup_root = str(tmp_path / "cg"),
                           allowed_cpus = list(range(16)))


def test_plan_reserves_host_cores_on_gpu_node(tmp_path):
    topo = dual_socket(tmp_path)
    plan = H.numa_hosts.plan_workers(topo, gpu_node = 1, host_cores = 1, threads = None, numa = True)
    w0, w1 = plan["workers"]
    assert w0["node"] == 0 and w0["mem_node"] == 0 and w0["threads"] == 4
    assert sorted(w0["cpus"]) == [0, 1, 2, 3, 8, 9, 10, 11]
    assert w1["node"] == 1 and w1["threads"] == 3
    assert 7 not in w1["cpus"] and 15 not in w1["cpus"]
    assert plan["host_cpus"] == [7, 15]
    assert not set(plan["host_cpus"]) & set(w0["cpus"] + w1["cpus"])


def test_plan_unknown_gpu_reserves_on_lowest_node(tmp_path):
    topo = dual_socket(tmp_path)
    plan = H.numa_hosts.plan_workers(topo, gpu_node = None, host_cores = 1, threads = 2, numa = True)
    assert plan["host_cpus"] == [3, 11]
    assert all(w["threads"] == 2 for w in plan["workers"])


def test_plan_single_node_or_off_is_one_unconfined_worker(tmp_path):
    make_sysfs(str(tmp_path), {0: [(c, c, 0) for c in range(4)]})
    topo = H.topo.discover(sysfs_root = str(tmp_path), cgroup_root = str(tmp_path / "cg"), allowed_cpus = [0, 1, 2, 3])
    for numa in (True, False):
        plan = H.numa_hosts.plan_workers(topo, None, 1, None, numa)
        assert len(plan["workers"]) == 1 and plan["workers"][0]["cpus"] is None and not plan["numa"]


def test_plan_respects_cpuset(tmp_path):
    nodes = {
        0: [(c, c, 0) for c in range(4)] + [(c + 8, c, 0) for c in range(4)],
        1: [(c, c - 4, 1) for c in range(4, 8)] + [(c + 8, c - 4, 1) for c in range(4, 8)],
    }
    make_sysfs(str(tmp_path), nodes)
    topo = H.topo.discover(sysfs_root = str(tmp_path), cgroup_root = str(tmp_path / "cg"),
                           allowed_cpus = [0, 1, 4, 5, 8, 12])
    plan = H.numa_hosts.plan_workers(topo, None, 0, None, True)
    allowed = {0, 1, 4, 5, 8, 12}
    for w in plan["workers"]:
        assert set(w["cpus"]) <= allowed
    assert plan["workers"][0]["threads"] == 2 and plan["workers"][1]["threads"] == 2


def test_shard_layouts(monkeypatch):
    g = object.__new__(H.numa_hosts.HostGroup)
    g.hosts = [object(), object()]
    monkeypatch.setenv("EXL3_NUMA_MEMORY_POLICY", "shard")
    monkeypatch.setenv("EXL3_NUMA_SHARD", "mod")
    assert g.shard(5) == [[0, 2, 4], [1, 3]]
    monkeypatch.setenv("EXL3_NUMA_SHARD", "blocked")
    assert g.shard(5) == [[0, 1, 2], [3, 4]]
    g.hosts = [object()] * 6                    # E = 8 over 6 nodes: no empty shard
    assert g.shard(8) == [[0, 1], [2, 3], [4], [5], [6], [7]]
    g.hosts = [object(), object()]
    monkeypatch.setenv("EXL3_NUMA_MEMORY_POLICY", "replicate")
    assert g.shard(3) == [[0, 1, 2], [0, 1, 2]]
    monkeypatch.setenv("EXL3_NUMA_MEMORY_POLICY", "bogus")
    with pytest.raises(ValueError):
        g.shard(3)


def test_gpu_node_unknown_stays_none(tmp_path):
    assert H.topo.gpu_numa_node(0, sysfs_root = str(tmp_path), proc_root = str(tmp_path), env = {},
                                bus_id = "0000:3b:00.0") is None


def many_nodes(tmp_path, n, mem_kb = None):
    nodes = {i: [(2 * i, 0, i), (2 * i + 1, 1, i)] for i in range(n)}
    make_sysfs(str(tmp_path), nodes, mem_kb)
    return H.topo.discover(sysfs_root = str(tmp_path), cgroup_root = str(tmp_path / "cg"),
                           allowed_cpus = list(range(2 * n)))


def test_plan_refuses_more_nodes_than_the_dispatcher_supports(tmp_path):
    topo = many_nodes(tmp_path, 9)
    with pytest.raises(RuntimeError, match = "at most 8"):
        H.numa_hosts.plan_workers(topo, None, 0, None, True, max_nodes = 8)
    # The classic per-layer path has no such limit
    assert len(H.numa_hosts.plan_workers(topo, None, 0, None, True)["workers"]) == 9


def test_memoryless_cpu_node_binds_to_the_nearest_memory(tmp_path):
    topo = many_nodes(tmp_path, 3, mem_kb = {0: 1048576, 1: 0, 2: 1048576})
    plan = H.numa_hosts.plan_workers(topo, None, 0, None, True)
    mem = {w["node"]: w["mem_node"] for w in plan["workers"]}
    assert mem[0] == 0 and mem[2] == 2 and mem[1] in (0, 2)


def test_pin_rule_matches_the_worker_pool(monkeypatch):
    monkeypatch.setattr(H.numa_hosts.sys, "platform", "linux")
    monkeypatch.delenv("EXL3_MOE_CPU_PIN", raising = False)
    assert not H.numa_hosts.pin_threads_enabled()
    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "1")
    assert H.numa_hosts.pin_threads_enabled()
    monkeypatch.setattr(H.numa_hosts.sys, "platform", "win32")
    monkeypatch.delenv("EXL3_MOE_CPU_PIN", raising = False)
    assert H.numa_hosts.pin_threads_enabled()
    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    assert not H.numa_hosts.pin_threads_enabled()
