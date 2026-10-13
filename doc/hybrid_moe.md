# Hybrid MoE: shared GPU expert cache, zero-copy homes, NUMA workers

Experimental, off by default. `-mh` / `--moe_hybrid` (or `EXL3_HYBRID_MOE=1`) runs every eligible
block-sparse MoE layer with its routed experts held in host RAM. Each selected expert is computed by
one of four lanes, chosen per layer call:

| Lane | Where the weights are read from | Math |
|---|---|---|
| Cache hit | a VRAM slot of the device's shared expert cache | the unmodified GPU MoE kernels |
| Admission | copied from the host home into a slot in the same call, then as a hit | the unmodified GPU MoE kernels |
| Zero-copy | the pinned, mapped host home, read over PCIe by the GPU kernels | the unmodified GPU MoE kernels |
| CPU | the CPU worker of one NUMA node (its own copy of the expert) | the existing int8-activation CPU kernels |

The first three lanes run the unmodified GPU kernels on the original EXL3 weights. The CPU lane is
the same approximation the existing `-mcl` / `-mcs` offload uses.
`EXL3_EXPERT_DISPATCH_POLICY=gpu_exact` keeps every expert with a GPU-readable home on the first
three lanes. Under `gpu_exact` a layer call is bit-identical to the all-VRAM model in these cases:

- decode-size calls, which use the coop and fused kernels;
- prefill calls in which no expert exceeds the batched tier's per-expert row cap.

Above that cap the all-VRAM model uses its per-expert dequant path. Hybrid mode uses single-expert
batched groups instead, which round differently in fp16 (see [Prefill](#prefill)).

Base: `turboderp-org/exllamav3` branch `dev` at `6cd89a908d957afba1cd1659fcf7d99a9e56d776`
(2026-10-09, version string 1.6.0; tag `v1.6.0` is `151539c`). Work branch:
`feat/hybrid-moe-numa-amx`.

Nothing in this document is a performance claim. None of it has run on an H100 or a dual-socket
Xeon host; see [Verification status](#verification-status).

## Contents

- [What upstream already had](#what-upstream-already-had)
- [Execution flow](#execution-flow)
- [GPU expert cache](#gpu-expert-cache)
- [Host homes and zero-copy](#host-homes-and-zero-copy)
- [NUMA workers](#numa-workers)
- [Dispatch policies](#dispatch-policies)
- [Prefill](#prefill)
- [Speculative decoding](#speculative-decoding)
- [Configuration](#configuration)
- [Telemetry](#telemetry)
- [Failure handling](#failure-handling)
- [Mechanism map](#mechanism-map)
- [Limitations](#limitations)
- [Verification status](#verification-status)
- [Changed files](#changed-files)

## What upstream already had

The extension reuses these and changes them only where a gap existed.

| Mechanism | Upstream location | Reused as | Gap closed here |
|---|---|---|---|
| CPU worker process, pinned job ring, stream memops, fused issue/collect | `model/moe_cpu_host.py`, `exllamav3_ext/cpu/moe_handoff.*`, `routing.cu` | the CPU lane of every node | one worker per NUMA node; node-bound memory |
| AVX2 / AVX-512 mul1 CPU kernels | `exllamav3_ext/cpu/moe_mul1.cpp` | unchanged | pool core order now follows the process affinity mask |
| Whole-layer (`-mcl`) and per-layer split (`-mcs`) offload | `modules/block_sparse_mlp_cpu.py` | unchanged (`legacy`) | with `EXL3_NUMA_ENABLE=1`, layers round-robin over per-node workers |
| `-mcs` dynamic hot/cold placement | `block_sparse_mlp_cpu.py` | unchanged | fixed slots per layer, re-reads the checkpoint: the hybrid cache replaces it in hybrid mode |
| memfd pinned arena (`PORTABLE`) | `moe_cpu_host.py` | the `worker_arena` home | registered `MAPPED` with a per-chunk fallback |
| Device pointer tables per expert | `modules/multilinear.py`, `BC_BlockSparseMLP` | the cache's control point | rows owned and rewritten by the cache |
| Fused / batched-reconstruct prefill tiers | `block_sparse_mlp.py`, `moe_batch_recon.py` | unchanged kernels | per-call fused row cap; bounded single-expert groups for leftovers |
| Speculative decoding (MTP, draft model, DFlash2, n-gram) | `generator/` | unchanged | none needed: verify forwards are ordinary multi-row forwards |

Smaller upstream inconsistencies found during the audit are listed under
[Limitations](#limitations). They are not fixed here unless they touched the new code.

## Execution flow

```
BlockSparseMLP.forward
  routing (unchanged)
  HybridDispatcher.submit                       model/moe_hybrid_dispatch.py
    moe_hybrid_issue  (1 block, compute stream) exllamav3_ext/moe_hybrid.cu
      count picks per expert; hits: CLOCK ref bit + pin epoch
      misses, hottest first: GPU quota -> admit (CLOCK victim) or zero-copy; rest -> CPU node
      masked selection for the GPU (CPU picks -> id num_experts)
      decode: sel / weights into each active node's pinned slot, slot gate
    moe_hybrid_stage_x (decode)                 activations into the active slots
    moe_hybrid_gather (decode, if admitting)    home -> slot copies of this call's admissions
    per node: data_ready memop                  worker starts
  GPU tiers on the masked selection (unchanged kernels, pointer tables)
  shared expert (overlaps the CPU work at decode)
  HybridDispatcher.combine                      per node: done wait, add partial, consumed
```

Every step of one device runs on that device's compute stream, in this order. Prefill sends the
CPU work through per-node side streams instead (see [Prefill](#prefill)).

## GPU expert cache

`model/moe_expert_cache.py`. There is one owner per CUDA device, and every hybrid layer on that
device registers with it: main model, MTP head and draft model alike. The owner holds the device's
only cache budget.

**Control point.** A hybrid layer's routed `Linear`s are built over a per-size bounce block, so
`load_local` creates its pointer tables and `BC_BlockSparseMLP` normally while no expert bytes go
to VRAM. The owner then controls the per-expert trellis row of every table that holds one:

- `MultiLinear.ptrs_trellis` of gate, up and down. These are read by the fused decode kernels on
  every launch, by `exl3_moe` per call, and by the batched reconstruct tier through `index_select`.
- The stacked gate/up table handed to `BC_BlockSparseMLP`. No kernel reads it today; it is kept
  consistent anyway.

The host copies `*_ptrs_trellis_cpu` inside `BC_BlockSparseMLP` are snapshots taken at
construction that no code reads. They are not updated. suh/svh stay real per-expert VRAM tensors.

**Replacement.** CLOCK runs over the slots of the layer's size class, shared by every layer of
every component on the device. The issue kernel marks each hit's reference bit and stamps the slot
with the call's epoch. Victim search skips disabled slots and slots stamped by the current call,
and clears reference bits on the way. Admission runs only for calls of at most
`EXL3_EXPERT_CACHE_ADMIT_MAX` picks (decode and small verify windows), at most
`EXL3_EXPERT_CACHE_GATHER_MAX` per call. Prefill only marks hits. An evicted expert's rows go back
to its home, or to the bounce block when the home is not GPU-readable.

**Why slot reuse is safe.** Admission, eviction and the copy into a slot are kernels on the compute
stream. Every MoE kernel that could still read the old tenant was enqueued earlier on the same
stream, so it has finished before the issue kernel runs. Within one call, the pin epoch keeps the
call's own hits and admissions out of victim search. Host-side edits follow three rules:

- **Edits that only enable or redirect** (enabling slots, warm installs, resetting a layer's rows)
  are enqueued on the current stream.
- **Edits that free memory** (shrinking, `release_all`, unregistering a layer) quiesce first. An
  event is recorded on every stream the owner has seen and on the current stream, and each is
  waited for. Then the edit runs, is waited for, and only then is memory released.
- **Homes are immutable** while registered. Eviction never frees host memory.

**Budget.** The cache is sized lazily, at the first forward pass of any hybrid component. By then
weights, KV cache, recurrent state and the statics of every loaded component exist. The size is
the memory actually available (`cudaMemGetInfo` free plus the caching allocator's
reserved-but-unallocated pool) minus a floor. The floor is the larger of:

- `EXL3_EXPERT_CACHE_RESERVE_MB`;
- the largest known transient plus `EXL3_EXPERT_CACHE_MIN_FREE_MB`. The transient comes from four
  places:
  - every autosplit load of any component on the device (main model, MTP head, vision tower,
    draft model), plus `EXL3_AUTOSPLIT_MARGIN_MB`. These are max-merged per device, so a later
    component never lowers them, whichever loading API it used. They are cleared when the device's
    last hybrid layer unloads, so a smaller model loaded next is measured afresh.
  - transients noted on the owner;
  - the quantized KV cache's prefill staging scratch (fp16 K and V of the whole cache pool, plus
    `EXL3_AUTOSPLIT_MARGIN_MB`), noted at each load's first forward. Autosplit loads already measure
    it; loads with an explicit `--gpu_split` would otherwise keep only the fixed reserve free for a
    scratch that every long prefill chunk allocates at full size;
  - the largest prefill MoE bound seen so far.

`EXL3_EXPERT_CACHE_LIMIT_GB` caps the size. Nothing is hard-coded to a card size.

The fixed reserve is a floor, not the safety mechanism:

- **Before every prefill-size pass** of a hybrid component, the owner checks the floor again,
  including that pass's worst-case MoE transient, and gives back whole chunks while it is short.
  The measured transient already contains the MoE bound at the load's chunk size. The pass's bound
  therefore raises the requirement only where it is larger, so nothing is counted twice. A vision
  encode or any other component's forward does not trigger this check.
- **When the generator goes idle,** the owner regrows in whole chunks while the floor plus
  `EXL3_EXPERT_CACHE_GROW_HYST_MB` stays free. Because the floor includes the largest prefill bound
  seen, regrowth stops short of it, and requests do not shrink and regrow the cache each time.
- **A failed chunk allocation** ends growth; it is never fatal.
- **Loading another model component** first releases every chunk on every device
  (`release_all_caches`), so that load sees the device as it would without the cache.

**CLOCK range.** The slot index layout covers the whole device or the limit, so that chunks can
come and go without moving a slot. The victim scan, though, covers only the enabled prefix of each
class. Growth enables the lowest free chunk and shrinking disables the highest one, so the enabled
chunks always form a prefix. Each grow and shrink writes that prefix's end into the class range,
stream-ordered.

**Size classes.** Experts of different byte sizes (another model, another quantization) get
separate slot classes. Each class has its own chunks and CLOCK hand. All classes share the one
device budget in proportion to the expert bytes registered in each:

- A weighted class whose share rounds down below one chunk still gets one chunk in the layout.
  If the limit cannot hold even that, a notice names the class.
- Growth skips a class whose next chunk does not fit and continues with the others.
- When a class's last layer leaves, its chunks are freed and its id becomes reusable.
- When the owner's last layer leaves, the classes, their bounce blocks and the per-load hints are
  dropped.

**Warm start and stats.**

- `EXL3_EXPERT_CACHE_WARM=<json>` fills the first chunks with the hottest experts per class. It
  runs once per owner generation: a later re-layout keeps the live pick history. The file maps layer
  keys (or `component:key`) to per-expert counts, the same format as `EXL3_MOE_CPU_SPLIT_STATS`.
- `EXL3_EXPERT_CACHE_SAVE_STATS=<json>` writes the decayed pick history, merged into the file,
  each time the last layer of a component unregisters, and at exit.
- **Layer ids** are `component:key`. A second live dispatcher with the same component name, such
  as a draft model loaded as `text`, gets `text#1`, so its layers never collide with the main
  model's. A dispatcher returns its tag when its last layer unloads, so a reload gets the plain name
  and the same warm-start and stats keys back.
- A stats file holds per-layer counts by these keys, so it belongs to one model. A file shared
  between models with equal keys and expert counts would warm-start one with the other's counts.

See [Telemetry](#telemetry) for the counters.

## Host homes and zero-copy

A home is a native-layout (GPU tile order) copy of an expert's gate, up and down trellis, back to
back in pinned, mapped host memory. GPU kernels read it through the device alias from
`cudaHostGetDevicePointer`. The host address is never assumed to be a valid device address.

Two modes, chosen explicitly:

- `EXL3_EXPERT_HOME=native_copy` (default). Each CPU worker writes a second, native copy of its
  experts into shared memfd chunks while it loads the checkpoint, so the checkpoint is read once.
  The chunks are bound to the GPU's NUMA node when that node is known. The parent maps and
  registers each 1 GiB chunk as it is published. The worker's own arena stays in the
  CPU-optimized packed layout. RAM cost: one extra copy of the experts. With replicated worker
  arenas, only one worker writes the native copy.
- `EXL3_EXPERT_HOME=worker_arena`. No extra copy: the worker's own memfd arena is registered
  mapped and serves as the home. GPU kernels only read native tile order, so this needs
  `EXL3_MOE_CPU_SWIZZLE=0`, which costs the CPU kernels their packed layout. With the packed layout
  on, the dispatcher states this once and gives non-resident experts CPU lanes only.

**Registration at scale.** Registering 100+ GiB is not guaranteed to work (`RLIMIT_MEMLOCK`, driver
limits). Each chunk is registered separately, inside the `EXL3_EXPERT_HOME_PIN_LIMIT_GB` budget. A
chunk whose registration fails stays mapped but pageable, and its experts get no GPU-readable home.
Such an expert runs on a CPU lane, or from a slot when a warm start installs it with a synchronous
host copy. After two consecutive failures no further chunk is attempted. Registered bytes are
logged every 8 GiB. Teardown unregisters registered chunks only, after the views are dropped.

**RAM-only operation.** The runtime never re-reads the checkpoint. Evicted experts fall back to
their resident home, and the CPU workers keep their own copies. The checkpoint may live on tmpfs
and is read once at load. This differs from the `-mcs` hot/cold swap, which re-reads both sides
of every swap from the checkpoint.

## NUMA workers

`util/numa_topology.py`, `util/numa_mem.py`, `model/moe_numa_hosts.py`. Enable with
`EXL3_NUMA_ENABLE=1`.

**Topology** comes from sysfs:

- per node: cpulist, distances, and meminfo (MemFree + Inactive(file) as available memory);
- per CPU: core and package ids;
- the effective cpuset (affinity ∩ cgroup `cpuset.cpus.effective`) and the cgroup memory limit.

`EXL3_NUMA_CPU_MAP="0:0-63,128-191;1:64-127,192-255"` overrides the node-to-CPU map and is
validated against the cpuset. A single CPU node, or a host without NUMA sysfs, keeps the classic
single worker.

**GPU locality** is resolved in this order:

1. `EXL3_NUMA_GPU_NODE=dev:node` (explicit);
2. the PCI device's `numa_node` in sysfs, using the bus id torch reports;
3. `/proc/driver/nvidia`, only when exactly one GPU is listed.

If none of these gives an answer, the node is unknown: GPU-adjacent buffers get default placement,
the reserved host cores go to the lowest node, and a notice says so.

**Workers.** Each CPU node gets one of the existing worker processes, with its own job ring, slots
and thread pool:

- confined to the node's allowed CPUs (the pool derives its core order from the mask);
- its arena bound to the node (`mbind`, plus `set_mempolicy` around `fallocate` for shared memory);
- the node is the worker's preferred node for everything else.

The host process keeps `EXL3_MOE_HOST_CORES` physical cores, both SMT siblings, on the GPU's node.
It is confined to them only when the worker pools pin their threads. The rule is the pool's own:
pinned unless `EXL3_MOE_CPU_PIN=0`. A plan with one
unconfined worker (single node, NUMA off) uses the classic host placement unchanged. The per-node
groups of the classic `-mcl` / `-mcs` path are placed after their workers start.

Memory checks and limits:

- Arena chunks, lazy ones included, are checked against the node's available memory as well as the
  global and cgroup limits. Lazy chunks are bound to the node too.
- A CPU node without memory binds its worker's memory to the nearest node that has memory, by the
  distance row. The GPU-adjacent buffers (native home copy, handoff ring) are remapped the same way
  when the GPU's node has no memory.
- Hybrid mode supports at most 8 worker nodes. A host with more CPU nodes gets an explicit error
  before any worker starts. The fix is to restrict the CPUs with `EXL3_NUMA_CPU_MAP` or a cpuset,
  or to set `EXL3_NUMA_ENABLE=0`. The classic per-layer path has no such limit.

**Memory locality.**

| Kind | Placement |
|---|---|
| CPU-local: worker arenas | the worker's node |
| GPU-local: job-ring segment and flags, native home copy | the GPU's node, only when that node is known |

**Expert placement.**

- `EXL3_NUMA_MEMORY_POLICY=shard` (default): each expert lives on exactly one node. With
  `EXL3_NUMA_SHARD=mod` that is node `e mod N`. With `blocked` it is contiguous ranges whose sizes
  differ by at most one, so no node is left empty while there are at least as many experts as nodes.
- `replicate`: every node holds every expert, so the CPU lane can pick a node per call. This costs
  N times the arena RAM and is checked against each node's memory.

Expert weights never move between nodes per token.

**Classic offload with NUMA.** With `-mcl` / `-mcs` and `EXL3_NUMA_ENABLE=1`, whole layers are
assigned round-robin to the per-node workers. Each layer then lives and runs on one node, and
nothing else changes.

## Dispatch policies

`EXL3_EXPERT_DISPATCH_POLICY`:

- `legacy`: the hybrid path is off; layers load as upstream.
- `gpu_exact`: misses go zero-copy or are admitted. CPU lanes are used only for experts with no
  GPU-readable home.
- `hybrid` (default): misses are ordered hottest first, by tokens in this call and then by decayed
  pick history.
  - **Fixed quota.** With `EXL3_MOE_ZC_MAX=<n>`, the first `n` misses that could take either lane go
    zero-copy or are admitted. The rest go to the CPU lane of the node that holds them.
  - **Cost model** (`auto`, the default). Experts with only one possible lane are placed first.
    Each remaining miss then goes to its CPU node only if that makes the call's slowest lane
    strictly shorter than keeping the miss on the GPU; a tie stays on the GPU. Under the measured
    costs, a CPU lane therefore never lengthens the layer's critical path. The slowest lane is
    the larger of two times:
    - GPU time: resident experts × `t_hit` plus GPU-lane misses × `t_zc`;
    - each CPU node's time: `t_cpu_fixed` once the node is used, plus `t_cpu` per expert.

    The kernel computes this in float32 with explicit round-to-nearest operations, and the
    reference reproduces it exactly. Every input is measured:
    - `t_zc`: one expert's bytes at the rate of a timed copy out of a registered home;
    - `t_hit`: one expert's bytes at the rate of a timed device-to-device copy;
    - `t_cpu_fixed`: the median round trip of an empty gated job to each node, worst node;
    - `t_cpu`: worker telemetry, compute time per distinct expert of a job, EMA, slowest node.
      Only decode-shaped jobs of at most 8 rows count; prefill jobs carry many rows per expert.

    Once the model keeps every flexible miss on the GPU, decode produces no new CPU samples. So
    after `EXL3_MOE_COST_REMEASURE` passes (default 256, `0` = off) without a sample, one pass
    uses the bootstrap share to measure the CPU again.

    Until `t_zc` and `t_cpu` are both measured, `EXL3_MOE_ZC_CPU_FRAC_INIT` sets the CPU share of
    the misses that could take either lane. No Xeon or PCIe latency is assumed anywhere. The model
    treats a GPU or CPU expert's cost as independent of its row count, which is accurate for decode.

`EXL3_NUMA_POLICY` chooses the node when an expert is held by several nodes (`replicate`):

- `numa_local`, `numa_sharded`: the GPU's node when known, else the first holder. With shards,
  the owner is the only holder.
- `numa_balanced`: the holder with the fewest assigned rows in this call.
- `numa_adaptive`: the holder minimizing (rows assigned + this expert's rows) × measured cost.

**Worker telemetry** is opt-in (`telemetry_offset` in `exl3_moe_cpu_worker_run`; off for every
upstream path). The worker writes compute time, assignments and distinct experts into 64 bytes at
the end of each slot before publishing `done`.

## Prefill

Calls wider than the decode boundary (`EXL3_EXPERT_DECODE_MAX_ROWS`, default and maximum the CPU
slot capacity `EXL3_MOE_CPU_SLOT_ROWS`) take the prefill path and mark hits only. A non-resident expert with a
GPU-readable home runs zero-copy when it has at least `EXL3_EXPERT_PREFILL_GPU_MIN` rows in the
chunk; otherwise it goes to its node's CPU lane. `gpu_exact` sends every readable expert to the GPU.

The CPU lanes' rows are compressed to tokens that have CPU work. They are issued on one side
stream per node, so the GPU tiers and every node run concurrently, and they are joined with events
before the add. Tensors used on a side stream are `record_stream`ed.

The GPU tiers are the upstream ones with two changes:

- **Fused row cap** (`hybrid` policy only). When any expert is read over PCIe in a call, the fused
  tier's row cap drops to `EXL3_EXPERT_CACHE_ZC_FUSED_ROWS` for that call. The fused kernel re-reads
  an expert's weights once per row tile; the batched reconstruct tier reads them once per call. The
  slot-scratch accounting uses the same cap. The cap moves resident experts to another tier too, so
  `gpu_exact` never applies it and keeps the all-VRAM tier plan at the cost of more PCIe reads.
- **Leftover experts.** An expert neither tier takes (above the batched tier's per-expert row cap)
  runs as single-expert batched groups of at most that cap, accumulating weighted. Every group
  stays inside the batched tier's own buffer bound, and its weights come from the live tables.
  `BlockSparseMLP._autosplit_worst_case` includes these groups and the CPU lanes' staging.

## Speculative decoding

Verify forwards from MTP heads, draft models, DFlash2 and n-gram drafting are forwards with
`jobs × (1 + window)` rows. They take the decode path up to `EXL3_EXPERT_DECODE_MAX_ROWS` rows
(default `EXL3_MOE_CPU_SLOT_ROWS`, 64) and the prefill path above it. Nothing in the dispatcher,
cache or workers assumes one row.

The boundary sits at the slot capacity rather than at `EXL3_MOE_STREAM_MIN_ROWS` (32) so that
hybrid suffix-match drafting (`-hdt`), whose rounds verify `jobs × (1 + hybrid_draft_tokens)` rows,
and batched speculative decoding keep the decode lane policy (measured cost model, no host sync per
layer) instead of the prefill rule that sends every non-resident expert with fewer than
`EXL3_EXPERT_PREFILL_GPU_MIN` rows to a CPU lane. Set `EXL3_EXPERT_DECODE_MAX_ROWS=31` for the former
boundary.

Each component (main, `mtp`, a draft model) gets its own dispatcher and worker group, and they
share each device's cache owner. DFlash2's drafter has no MoE. Drafters and acceptance rules are
unchanged.

## Configuration

All new behaviour is off unless `-mh` / `EXL3_HYBRID_MOE=1` (or, for the NUMA workers of the
classic paths, `EXL3_NUMA_ENABLE=1`) is set.

| Variable | Default | Meaning |
|---|---|---|
| `EXL3_HYBRID_MOE` / `-mh` | `0` | hybrid MoE on eligible layers |
| `EXL3_EXPERT_DISPATCH_POLICY` | `hybrid` | `legacy`, `gpu_exact`, `hybrid` |
| `EXL3_EXPERT_CACHE_POLICY` | `clock` | `clock`; `none` = no slots (zero-copy and CPU lanes only) |
| `EXL3_EXPERT_CACHE_LIMIT_GB` | unset | hard cap of the cache per device |
| `EXL3_EXPERT_CACHE_RESERVE_MB` | `1536` | minimum free VRAM the cache leaves |
| `EXL3_EXPERT_CACHE_MIN_FREE_MB` | `1024` | added to the measured transient for the floor |
| `EXL3_EXPERT_CACHE_GROW_HYST_MB` | `256` | hysteresis for regrowth when idle |
| `EXL3_EXPERT_CACHE_ADMIT_MAX` | `64` | largest call (picks) that may admit |
| `EXL3_EXPERT_CACHE_GATHER_MAX` | `64` | admissions per call |
| `EXL3_EXPERT_CACHE_GATHER_BLOCKS` | `16` | blocks per admission copy |
| `EXL3_EXPERT_CACHE_CHUNK_SLOTS` | `32` | slots per allocation chunk |
| `EXL3_EXPERT_CACHE_MAX_LAYERS` | `256` | hybrid layers per device |
| `EXL3_EXPERT_CACHE_DECAY` | `0.5` | pick-history decay per idle period |
| `EXL3_EXPERT_CACHE_WARM` | unset | warm-start stats (json) |
| `EXL3_EXPERT_CACHE_SAVE_STATS` | unset | write stats (json) at unload / exit |
| `EXL3_EXPERT_CACHE_ZC_FUSED_ROWS` | `32` | fused-tier row cap in calls with zero-copy experts |
| `EXL3_EXPERT_CACHE_DEBUG` | `0` | print growth / shrink / warm start |
| `EXL3_EXPERT_HOME` | `native_copy` | `native_copy` or `worker_arena` |
| `EXL3_EXPERT_HOME_PIN_LIMIT_GB` | unset | registered-bytes cap for homes |
| `EXL3_EXPERT_DECODE_MAX_ROWS` | slot rows (`64`) | widest call on the decode path; wider calls take the prefill path |
| `EXL3_EXPERT_PREFILL_GPU_MIN` | `8` | prefill rows for a non-resident expert to go zero-copy |
| `EXL3_MOE_ZC_MAX` | `auto` | GPU quota of misses per decode call |
| `EXL3_MOE_ZC_CPU_FRAC_INIT` | `0.5` | CPU share of flexible misses until the cost model is measured |
| `EXL3_MOE_ADAPTIVE_EMA` | `0.1` | smoothing of measured CPU cost |
| `EXL3_MOE_COST_REMEASURE` | `256` | passes without a CPU cost sample before one pass re-measures (`0` = never) |
| `EXL3_HYBRID_MOE_DEBUG` | `0` | print the cost probes (PCIe, VRAM, CPU round trip) |
| `EXL3_HYBRID_MOE_PROFILE` | `0` | GPU-side CPU-lane wait profile with CUDA events (see [Telemetry](#telemetry)) |
| `EXL3_NUMA_ENABLE` | `0` | one CPU worker per NUMA node (hybrid: at most 8) |
| `EXL3_NUMA_POLICY` | `numa_local` | `numa_local`, `numa_sharded`, `numa_balanced`, `numa_adaptive` |
| `EXL3_NUMA_MEMORY_POLICY` | `shard` | `shard` or `replicate` |
| `EXL3_NUMA_SHARD` | `mod` | `mod` or `blocked` |
| `EXL3_NUMA_CPU_MAP` | auto | node:cpulist override, `;`-separated |
| `EXL3_NUMA_GPU_NODE` | auto | `dev:node[,dev:node]` declared GPU locality |
| `EXL3_NUMA_DEBUG` | `0` | print topology and worker placement |

Existing knobs keep their meaning: `-mct` / `EXL3_MOE_CPU_THREADS` (threads per worker),
`EXL3_MOE_CPU_PIN`, `EXL3_MOE_HOST_CORES`, `EXL3_MOE_CPU_SLOT_ROWS`, `EXL3_MOE_STREAM_MIN_ROWS`,
`EXL3_MOE_CPU_SWIZZLE`, `EXL3_MOE_ARENA_HUGEPAGE`, `EXL3_MOE_ARENA_HUGE`,
`EXL3_HOST_MEM_RESERVE_MB`.

**Supported combinations.**

- **Cache only:** `-mh`.
- **NUMA only:** `EXL3_NUMA_ENABLE=1` with `-mcl` / `-mcs`.
- **Cache and NUMA:** `-mh` with `EXL3_NUMA_ENABLE=1`.

Hybrid needs layer-split loading (no `-tp`) and excludes `-mcl` / `-mcs` for the same model.

**Eligibility** is checked per layer from headers only; ineligible layers load as usual. A layer
qualifies when it has:

- mul1-codebook experts stored as separate EXL3 tensors of one shape per projection, no
  per-expert biases, and no padded hidden width;
- a silu/gelu gated or relu2 gateless activation;
- at most 512 experts, and at least as many experts as worker nodes;
- no tensor-parallel or latent-MoE structure;
- the batched reconstruct tier available.

## Telemetry

Counters are kept on the device by the issue kernel, at the cost of a few integer adds per layer
call. They are read only on request, by `ExpertCache.summary()` or `HybridDispatcher.summary()`.
That read is one device-to-host copy, meant for use between passes and never per token.

| Counter | Meaning |
|---|---|
| `hits`, `misses`, `hit_rate` | unique (layer call, expert) pairs found resident or not |
| `admits`, `admit_bytes` | admissions and the bytes copied into slots over PCIe |
| `evictions`, `no_victim` | tenants displaced; admissions that found no victim (expert stayed zero-copy) |
| `zero_copy`, `zero_copy_bytes` | experts computed from host homes; their bytes, a lower bound on PCIe reads |
| `cpu`, `cpu_by_node` | experts sent to CPU lanes, total and per worker node |
| `no_lane` | picks with no lane at all: a configuration error, raised at the next idle point |
| `calls`, `prefill_calls` | issue calls |
| `slots_enabled`, `slots_occupied`, `slot_utilization` | slot usage |
| `cache_mib`, `floor_mib`, `vram_limit_mib`, `vram_available_mib`, `vram_total_mib` | budget state |

`HybridDispatcher.summary()` adds three things:

- the measured lane costs and whether the cost model is active;
- each node's CPU time per expert;
- with `EXL3_HYBRID_MOE_PROFILE=1`, the exposed CPU-lane wait per node (total, mean, maximum).

The wait is the GPU time from just before the done-flag wait to just after the partial is folded
in (decode), or the wait for the node's side stream (prefill). It is measured with CUDA event
pairs, which are harvested at the next pass start with `Event.query()` and never synchronized. At
most 512 pairs are pending; further calls are counted as unsampled.

## Failure handling

| Failure | Behaviour |
|---|---|
| `cudaHostRegister` fails for a home chunk | non-fatal: the chunk stays mapped but pageable, and its experts get no GPU-readable home (CPU lane, or a slot through a warm install). A notice names the chunk; after two consecutive failures no further chunk is tried. |
| No mapped memory (`worker_arena` with the packed CPU layout, or pin limit reached) | stated once; those experts take CPU lanes |
| An expert with neither a GPU-readable home nor a CPU copy | load-time error naming the experts; at run time `no_lane` is counted and raised at the next idle point |
| VRAM OOM while growing the cache | growth stops with a notice; never fatal. Prefill headroom is restored by shrinking before the pass |
| OOM while loading a hybrid layer (autosplit retry) | the layer's worker registrations are undone before the error propagates, so the retry on the next device starts clean |
| GPU NUMA node unknown | never guessed: GPU-adjacent buffers keep default placement, with a notice |
| More than 8 CPU NUMA nodes (hybrid) | explicit error before any worker starts |
| Memoryless CPU node | its worker's memory binds to the nearest node with memory |
| Node memory short | explicit error before the chunk is allocated, with options that exist |
| Worker process dies or times out | the watchdog sets the abort flag, then releases the GPU's flag waits so no stream hangs. The next layer call's reserve raises, and so does every later pass start. A forward already in flight may have folded in stale CPU partials; the generator checks the flag after each iteration's host synchronization and raises instead of returning that iteration's results. |
| Job ring full | the existing throttle waits for the worker; it raises if the worker aborts or dies meanwhile |
| A decode call larger than the CPU slot | explicit error from the binding (and the reference) before any write: rows × width and picks are checked against each node's slot capacity |
| Generation cancelled | nothing to undo: no host-side state spans layer calls, and slots and tables stay consistent at every kernel boundary |
| Reload after unload | the dispatcher's per-load state resets with its last layer; layers finalize only once their workers have started and acknowledged them |

## Mechanism map

| Mechanism | Source | Licence | How it appears here |
|---|---|---|---|
| Table-driven shared CLOCK cache over ExLlamaV3's per-expert pointer tables; dedup, mark hits, pick victims, repoint, gather copy | glm53-flash-offload `glm53/expert_cache.py` (0xSero) | MIT | Re-implemented natively. Owner per device, size classes, pin epoch, quiesced shrink. No monkeypatching. |
| Zero-copy reads of non-resident experts through mapped host memory | glm53-flash-offload `glm53/exl3_tiers.py` | MIT | Homes are native-layout memfd chunks registered `PORTABLE\|MAPPED`, with a per-chunk fallback. The device alias is always queried. |
| Elastic slots lent to prefill, re-taken for decode | glm53-flash-offload `expert_cache.py` | MIT | Generalized to a dynamic floor: shrink before passes, regrow when idle, release before component loads. |
| Min-max split of misses between a PCIe lane and a CPU lane | glm53-flash-offload `kernels/cpu_avx2/ft_tier_cu.cu`, `nv2_host.cpp` | MIT | Same objective. Inputs are measured (PCIe probe, worker telemetry), not calibrated constants. |
| Device-side wait on a mapped flag, no host sync per layer | glm53-flash-offload and upstream ExLlamaV3 | MIT | Upstream's memop handoff, unchanged. |
| Per-NUMA worker pools, thread pinning inside the node, node-bound memory via first touch | KTransformers kt-kernel `cpu_backend/worker_pool.*` (KVCache.AI) | Apache-2.0 | One existing ExLlamaV3 worker process per node. No code copied. |
| Expert sharding across NUMA nodes | KTransformers `operators/moe-tp.hpp` (splits the intermediate dim) | Apache-2.0 | Whole experts are sharded instead. EXL3 tiles and 128-point Hadamard blocks make an intermediate split invasive. |
| AMX int8/int4 MoE kernels | KTransformers `operators/amx/*` | Apache-2.0 | Not ported. An exact AMX tier over the EXL3 trellis was written instead (see Limitations). |
| Worker job ring, pinned arena, CPU kernels, prefill tiers, pointer tables | ExLlamaV3 | MIT | Reused unchanged except as listed in [Changed files](#changed-files). |

No source file was copied from either reference project. Everything above was written for this
codebase from the documented mechanisms.

## Limitations

- **Not run on GPU hardware.** The CUDA kernels compile for sm_90 (see
  [Verification status](#verification-status)) but have not been executed.
  `tests/kernels/moe/test_hybrid_kernels_parity.py` compares them against the reference on a GPU
  host, including the cost model and float32 node-cost ties, and has not run yet. Correct
  behaviour on a GPU is therefore not established.
- **Cost model inputs are probes.** `t_hit` and `t_zc` come from bulk copies and `t_cpu_fixed` from
  an empty job. Real kernel timings can differ, for example through PCIe read efficiency of the
  fused kernels or L2 effects. `EXL3_MOE_ZC_MAX=<n>` bypasses the model.
- **Single-node CI only.** The NUMA code ran on a one-node VM: `mbind`, `set_mempolicy` and the
  page-placement query work there. Dual-socket behaviour is covered by synthetic sysfs trees only.
- **CPU lanes are approximate.** They inherit the int8-activation error of the existing CPU
  kernels (about 0.9 % output RMS per call; e2e KL ≤ 1e-2 in upstream tests).
  `EXL3_EXPERT_DISPATCH_POLICY=gpu_exact` avoids them except for experts without a GPU-readable
  home.
- **Zero-copy cost.** A zero-copy read costs PCIe bandwidth per kernel pass. The fused decode
  kernel reads each (token, expert) run once. In prefill the fused-tier cap limits re-reads.
- **Admission and decode-size calls.** Admission uses SM-driven copies of mapped memory, not the
  DMA engine. It applies only to calls of at most `EXL3_EXPERT_CACHE_ADMIT_MAX` picks.
- **Prefill sync.** Prefill syncs the host once per hybrid layer to plan the CPU lanes. The GPU
  tiers already sync on their counts.
- **`native_copy` RAM.** It doubles host RAM for experts (about 114 GB extra for GLM-5.3-Flash at
  3.05 bpw), and `replicate` adds N × arena.
- **Eligibility.** Layers with per-expert biases, padded widths, fused expert tensors, latent MoE
  or tensor parallelism stay on the regular path.
- **AMX tier: exact, CPU-validated, not run on the target host.** On CPUs with AMX-INT8 (Sapphire
  Rapids and later; Linux 5.16+ for the tile-state permission) the CPU expert kernel adds an AMX
  pass to the `vnni`/`vbmi` tier (`amx_tiles` in `cpu/moe_mul1.cpp`):
  - **Exact.** `TDPBSUD` with the activations as A (signed; `splat32` is already 16 k-rows of x8
    replicated ×4 per GEMV row) and the 16 decoded product rows of one trellis tile as B
    (unsigned) gives C[row][n] = the VNNI accumulator. The fp32 epilogue is shared, so outputs are
    bit-identical to the vector tier at every rate, layout, chunk size and thread count
    (`tests/moe_cpu/test_amx.py`, wide rows included).
  - **Where it pays.** Each weight tile is decoded once per 16 GEMV rows instead of once per 4, so
    the tier takes chunks of up to 16 tokens per expert and uses the tile kernel for passes of at
    least `EXL3_MOE_CPU_AMX_MIN_ROWS` (default 4) rows. Decode (one row per expert) is unchanged.
    On a 4-vCPU Sapphire Rapids VM, GLM-5.3-Flash-sized experts (4096 → 2048, 8 experts × 64
    tokens, 4 threads) ran 3.4–4.8× faster per token than the `vbmi` tier at K2–K4.
  - **Dispatch.** Prefill still sends non-resident experts with at least
    `EXL3_EXPERT_PREFILL_GPU_MIN` rows to zero-copy reads, so the CPU lanes see few-row experts
    unless that threshold is raised. Choosing the lane per expert from measured costs in prefill,
    as decode does, is the follow-up.
  - **Not used:** KTransformers' AMX kernels, which consume their own requantized int8/int4
    layouts and cannot read EXL3 trellis tensors.
- **Audit notes (upstream, unchanged here).**
  - The docstring of `_split_sweep_layer` says demotion is a map update, but it re-reads the
    checkpoint.
  - `moe_cpu_split_assigned` is never decremented on unload.
  - The `EXL3_MOE_CPU_SPLIT*` / `EXL3_MOE_CPU_SWAP*` knobs are undocumented (now listed in
    `env_vars.md`).
  - `env_vars.md` stated a stream threshold base of 16 where the code uses 8 (corrected).
  - The flag-order comment in `moe_handoff.h` omitted `consumed` (corrected).

## Verification status

| Check | Where | Result |
|---|---|---|
| Cache owner and policy (reference kernels, real memory): registration, CLOCK with pin epoch, gather bytes, lanes, node policies, prefill threshold, shrink/regrow, size classes, unregistration, warm start, registration fallback | `tests/moe_cpu/test_hybrid_cache.py` | passes on CPU |
| Hardening: CLOCK range on the enabled prefix, 400-step interleaving of routing with shrink / regrow / release / unregister over three size classes and three components (every GPU lane reads its expert's correct bytes), speculative verify windows, slot-capacity guard (segfault regression), absent node columns, float32 tie parity, cost model never lengthening the slowest lane (randomized), ties to the GPU, per-node counters, headroom without churn, device transients, stats per component, class reuse, small limits, growth across classes, warm start once | `tests/moe_cpu/test_hybrid_cache.py` | passes on CPU |
| Dispatcher with fake workers: rows 1/8/9/31/32/64/128/256/512 (path choice, admission cap, CPU partials folded in on both paths, prefill path on CPU), `gpu_exact` fused plan, absent node, component tags, reset on last detach, deferred finalize, load-failure cleanup, worst-case bound, measured cost inputs, telemetry harvest, worker death and oversized jobs | `tests/moe_cpu/test_hybrid_dispatch.py` | passes on CPU |
| NUMA planning: reserved cores, cpuset, shard layouts (balanced blocked), unknown GPU node, more than 8 nodes, memoryless nodes, pin rule | `tests/moe_cpu/test_hybrid_numa_plan.py` | passes on CPU |
| Topology discovery and memory policy | `tests/moe_cpu/test_numa_topology.py` | passes on CPU |
| Worker loop with telemetry off (upstream layout) and on: outputs bit-identical to `exl3_moe_cpu_forward`, records correct; pool core order follows `taskset` | CPU build of `cpu/moe_mul1.cpp` + host side of `cpu/moe_handoff.cu`, driven through `tests/testlib/moe_handoff.Segment` (scratch harness, not committed) | passes on CPU |
| CUDA sources compile for sm_90 (host + device, PTX generated for `moe_hybrid.cu`) | clang CUDA frontend, CUDA 12.8 headers, torch 2.14 headers | passes |
| Whole extension builds, links and imports for sm_90 (all 192 translation units, the repo's own JIT build in `exllamav3/ext.py`); new bindings present with the new signatures; no unresolved internal symbols | nvcc 13.0.88 (pip wheels), torch 2.14.1+cu130, gcc 13.3, no GPU | passes at the final commit |
| Every no-GPU test of `tests/moe_cpu` and `tests/kernels/moe_cpu_host/test_moe_worker.py` against the real extension, upstream tests included | same build | 185 pass, 30 skipped (need a GPU, a model, Windows or the slow marker) |
| CPU worker code under AddressSanitizer and UndefinedBehaviorSanitizer: worker loop with telemetry off and on, pool barrier stress, 40 random `exl3_moe_cpu_forward` shapes (rows 1 to 64, inactive picks) with outputs identical across thread counts | `cpu/moe_mul1.cpp` and the host side of `cpu/moe_handoff.cu` built with `-fsanitize=address,undefined -O1` | no reports |
| Kernel against reference, admitted slot bytes | `tests/kernels/moe/test_hybrid_kernels_parity.py` | GPU host only, not run |
| End-to-end parity with the all-GPU load (`hybrid`, `hybrid_exact` placements) | `tests/e2e/test_moe_offload.py` | GPU host with models only, not run |
| Speed | none | not measured; no claims |

**Earlier segfault (test run of the first commit).** `test_cpu_lanes_quota_and_staging` crashed
the interpreter.

- **Cause:** a test bug. The helper defaulted the row count to the number of picks, so `y` had
  four rows while the fake node slot held one. The reference then wrote four rows of
  activations into a one-row buffer, which corrupted the heap.
- **Underlying gap:** production had the same class of error. A decode call with more rows than
  the worker slot holds was refused only by an `assert` in `fused_reserve`, and the kernel binding
  could not check it at all.
- **Fix:** the node descriptor now carries each slot's activation and selection capacity. The
  binding and the reference both refuse an oversized call before writing anything.
  `fused_reserve` raises a `ValueError` instead of asserting.
- **Regression test:** `test_capacity_guard_rejects_oversized_decode_jobs`.

## Changed files

**New:**

- `exllamav3/exllamav3_ext/moe_hybrid.cu`, `moe_hybrid.cuh`
- `exllamav3/model/moe_expert_cache.py`, `moe_hybrid_ref.py`, `moe_hybrid_dispatch.py`, `moe_numa_hosts.py`
- `exllamav3/util/numa_topology.py`, `numa_mem.py`
- `tests/moe_cpu/test_hybrid_cache.py`, `test_hybrid_dispatch.py`, `test_hybrid_numa_plan.py`, `test_numa_topology.py`
- `tests/kernels/moe/test_hybrid_kernels_parity.py`
- `tests/testlib/hybrid_isolated.py`
- this document

**Modified:**

- `exllamav3/exllamav3_ext/bindings.cpp`: the hybrid kernels, and `telemetry_offset` on `exl3_moe_cpu_worker_run` (default -1).
- `exllamav3/exllamav3_ext/cpu/moe_handoff.h`, `moe_handoff.cu`: opt-in telemetry; flag-order comment.
- `exllamav3/exllamav3_ext/cpu/moe_mul1.cpp`: core order restricted to the process affinity mask.
- `exllamav3/model/moe_cpu_host.py`: node parameters, native home arena, mapped registration with fallback, fused reserve/publish/collect, telemetry, home teardown.
- `exllamav3/modules/block_sparse_mlp.py`, `block_sparse_mlp_cpu.py`: hybrid load, attach and forward; per-call fused cap; bounded leftover groups; NUMA routing of classic offload layers.
- `exllamav3/model/model.py`, `model_ls.py`, `config.py`, `exllamav3/model_init.py`, `exllamav3/generator/generator.py`: hooks, flag, CLI.
- `doc/env_vars.md`, `README.md`.
