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

The first three lanes are numerically identical to the all-VRAM model. The CPU lane is the same
approximation the existing `-mcl` / `-mcs` offload uses. `EXL3_EXPERT_DISPATCH_POLICY=gpu_exact`
keeps every expert on the first three lanes.

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
- the largest transient the autosplit loader measured on the device, plus
  `EXL3_AUTOSPLIT_MARGIN_MB`, plus `EXL3_EXPERT_CACHE_MIN_FREE_MB`.

`EXL3_EXPERT_CACHE_LIMIT_GB` caps the size. Nothing is hard-coded to a card size.

The fixed reserve is a floor, not the safety mechanism. Before every prefill-size pass the owner
re-checks the floor plus that pass's worst-case MoE transient, and gives back whole chunks while it
is short. When the generator goes idle it regrows one chunk at a time, keeping
`EXL3_EXPERT_CACHE_GROW_HYST_MB` of hysteresis. A failed chunk allocation ends growth; it is never
fatal. Loading another model component first releases every chunk on every device
(`release_all_caches`), so that load sees the device as it would without the cache.

**Size classes.** Experts of different byte sizes (another model, another quantization) get
separate slot classes. Each class has its own chunks and CLOCK hand. All classes share the one
device budget in proportion to the expert bytes registered in each.

**Warm start and stats.**

- `EXL3_EXPERT_CACHE_WARM=<json>` fills the first chunks with the hottest experts per class. The
  file maps layer keys (or `component:key`) to per-expert counts, the same format as
  `EXL3_MOE_CPU_SPLIT_STATS`.
- `EXL3_EXPERT_CACHE_SAVE_STATS=<json>` writes the decayed pick history at unload and at exit.
- `ExpertCache.summary()` reports hits, misses, admissions, evictions, zero-copy and CPU counts,
  and the slots in use.

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
It is confined to them only when `EXL3_MOE_CPU_PIN=1`, as upstream. Arena chunks are checked
against the node's available memory, as well as the global and cgroup limits.

**Memory locality.**

| Kind | Placement |
|---|---|
| CPU-local: worker arenas | the worker's node |
| GPU-local: job-ring segment and flags, native home copy | the GPU's node, only when that node is known |

**Expert placement.**

- `EXL3_NUMA_MEMORY_POLICY=shard` (default): each expert lives on exactly one node. With
  `EXL3_NUMA_SHARD=mod` that is node `e mod N`; with `blocked`, contiguous halves.
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
  pick history. The first ones up to the GPU quota go zero-copy or are admitted; the rest go to the
  CPU lane of the node that holds them. The quota is `EXL3_MOE_ZC_MAX` misses per call, or with
  `auto` a measured share. The share sent to the CPU is `t_zc / (t_zc + t_cpu)`:
  - `t_zc` is one expert's bytes at the PCIe rate measured by a timed copy out of a registered home;
  - `t_cpu` combines the nodes' rates, each node's cost per assignment coming from worker telemetry.

  Until both are measured, `EXL3_MOE_ZC_CPU_FRAC_INIT` applies. No Xeon or PCIe latency is
  assumed anywhere.

`EXL3_NUMA_POLICY` chooses the node when an expert is held by several nodes (`replicate`):

- `numa_local`, `numa_sharded`: the GPU's node when known, else the first holder. With shards,
  the owner is the only holder.
- `numa_balanced`: the holder with the fewest assigned rows in this call.
- `numa_adaptive`: the holder minimizing (rows assigned + this expert's rows) × measured cost.

**Worker telemetry** is opt-in (`telemetry_offset` in `exl3_moe_cpu_worker_run`; off for every
upstream path). The worker writes compute time, assignments and distinct experts into 64 bytes at
the end of each slot before publishing `done`.

## Prefill

Calls of at least `EXL3_MOE_STREAM_MIN_ROWS` rows mark hits only. A non-resident expert with a
GPU-readable home runs zero-copy when it has at least `EXL3_EXPERT_PREFILL_GPU_MIN` rows in the
chunk; otherwise it goes to its node's CPU lane. `gpu_exact` sends every readable expert to the GPU.

The CPU lanes' rows are compressed to tokens that have CPU work. They are issued on one side
stream per node, so the GPU tiers and every node run concurrently, and they are joined with events
before the add. Tensors used on a side stream are `record_stream`ed.

The GPU tiers are the upstream ones with two changes:

- **Fused row cap.** When any expert is read over PCIe in a call, the fused tier's row cap drops to
  `EXL3_EXPERT_CACHE_ZC_FUSED_ROWS` for that call. The fused kernel re-reads an expert's weights
  once per row tile; the batched reconstruct tier reads them once per call. The slot-scratch
  accounting uses the same cap.
- **Leftover experts.** An expert neither tier takes (above the batched tier's per-expert row cap)
  runs as single-expert batched groups of at most that cap, accumulating weighted. Every group
  stays inside the batched tier's own buffer bound, and its weights come from the live tables.
  `BlockSparseMLP._autosplit_worst_case` includes these groups and the CPU lanes' staging.

## Speculative decoding

Verify forwards from MTP heads, draft models, DFlash2 and n-gram drafting are forwards with
`jobs × (1 + window)` rows. They take the decode path (fewer rows than the streaming threshold, up
to `EXL3_MOE_CPU_SLOT_ROWS`) or the prefill path. Nothing in the dispatcher, cache or workers
assumes one row.

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
| `EXL3_EXPERT_PREFILL_GPU_MIN` | `8` | prefill rows for a non-resident expert to go zero-copy |
| `EXL3_MOE_ZC_MAX` | `auto` | GPU quota of misses per decode call |
| `EXL3_MOE_ZC_CPU_FRAC_INIT` | `0.5` | CPU share of misses before measurements exist |
| `EXL3_MOE_ADAPTIVE_EMA` | `0.1` | smoothing of measured CPU cost |
| `EXL3_HYBRID_MOE_DEBUG` | `0` | print the PCIe probe |
| `EXL3_NUMA_ENABLE` | `0` | one CPU worker per NUMA node |
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
| AMX int8/int4 MoE kernels | KTransformers `operators/amx/*` | Apache-2.0 | Not ported (see Limitations). |
| Worker job ring, pinned arena, CPU kernels, prefill tiers, pointer tables | ExLlamaV3 | MIT | Reused unchanged except as listed in [Changed files](#changed-files). |

No source file was copied from either reference project. Everything above was written for this
codebase from the documented mechanisms.

## Limitations

- **Not run on GPU hardware.** The CUDA kernels were compile-checked for sm_90 (clang CUDA,
  host and device, with PTX generation) against CUDA 12.8 headers. They have not been executed.
  `tests/kernels/moe/test_hybrid_kernels_parity.py` compares them against the reference on a GPU
  host and has not run yet.
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
- **AMX not implemented (Stage 2).** Stage 2 was deferred so it could not delay the cache,
  zero-copy, NUMA and dispatcher work. No AMX code or switch is included, and `EXL3_KT_AMX_ENABLE`
  does not exist yet. The analysis so far:
  - **Formats.** KTransformers' AMX kernels consume its own requantized int8/int4 layouts (per-row
    scales, VNNI-packed B tiles); they cannot read EXL3 trellis tensors. Using them would mean
    re-quantizing the model, which changes numerics and costs about 1 byte per weight of extra RAM.
    That option was rejected.
  - **Exact tier.** The existing VNNI kernel computes `bytesum(state * 0x83DCD12D) * x8` as
    `vpdpbusd`: the four product bytes are the unsigned operand, the int8 activation replicated ×4
    the signed one. `TDPBUSD` computes the same u8 × s8 dot products into int32 without
    saturation. A tier could therefore decode one 16×16 trellis tile into a 16-row × 64-byte A tile
    of product bytes, use the int8 activations of up to 16 token rows as the B tile, and accumulate
    the same exact integer sums. With the shared fp32 epilogue it would then match the VNNI tier
    bit for bit.
  - **Conditions for that claim**, which must be shown, not assumed:
    - no i32 overflow, with the existing bound of k ≤ 8192;
    - identical activation quantization, including wide rows (one row becomes two int8 rows);
    - the identical epilogue order.

    The AVX2/BW tiers emulate the byte-sum with saturating `vpmaddubsw` in a form documented as
    exact, so they are equal too.
  - **Value is low in this dispatch design.** AMX only pays at many token rows per expert. The
    hybrid dispatcher sends experts with at least `EXL3_EXPERT_PREFILL_GPU_MIN` rows to the GPU
    and keeps the CPU lanes for cold, few-row experts. A future AMX tier needs AMX hardware to
    validate against an emulated `TDPBUSD`.
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
| Dispatcher decode path with fake workers: masking, slot staging, combine, measured CPU share | `tests/moe_cpu/test_hybrid_dispatch.py` | passes on CPU |
| NUMA planning: reserved cores, cpuset, shard layouts, unknown GPU node | `tests/moe_cpu/test_hybrid_numa_plan.py` | passes on CPU |
| Topology discovery and memory policy | `tests/moe_cpu/test_numa_topology.py` | passes on CPU |
| Worker loop with telemetry off (upstream layout) and on: outputs bit-identical to `exl3_moe_cpu_forward`, records correct; pool core order follows `taskset` | CPU build of `cpu/moe_mul1.cpp` + host side of `cpu/moe_handoff.cu`, driven through `tests/testlib/moe_handoff.Segment` (scratch harness, not committed) | passes on CPU |
| CUDA sources compile for sm_90 (host + device, PTX generated for `moe_hybrid.cu`) | clang CUDA frontend, CUDA 12.8 headers, torch 2.14 headers | passes |
| Kernel against reference, admitted slot bytes | `tests/kernels/moe/test_hybrid_kernels_parity.py` | GPU host only, not run |
| End-to-end parity with the all-GPU load (`hybrid`, `hybrid_exact` placements) | `tests/e2e/test_moe_offload.py` | GPU host with models only, not run |
| Speed | none | not measured; no claims |

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
