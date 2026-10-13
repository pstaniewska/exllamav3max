#pragma once

#include <ATen/Tensor.h>
#include <cstdint>

// Hybrid MoE expert dispatch: one shared, table-driven GPU expert cache per device (CLOCK
// replacement over fixed-size VRAM slots), zero-copy reads of non-resident experts from a
// pinned + mapped native-layout host home, and CPU lanes on one or more (NUMA) worker processes.
// See exllamav3/model/moe_expert_cache.py (owner, state layout) and
// exllamav3/model/moe_hybrid_dispatch.py (per-component dispatcher). Everything here runs on
// the caller's current stream; correctness of slot reuse rests on stream order (an admission
// can only overwrite a slot after every earlier kernel on the stream has finished reading it)
// plus the per-call pin epoch (a call never evicts a slot it hits or admits itself).

#define HY_EMAX 512              // max experts per layer (= MAX_EXPERTS of BC_BlockSparseMLP)
#define HY_MAX_TABLES 6          // pointer tables per layer that hold per-expert trellis addresses
#define HY_MAX_NODES 8           // CPU worker nodes
#define HY_MAX_CLASSES 8         // slot size classes

// Indices into the state descriptor (int64 CPU tensor of addresses / values), mirrored by
// moe_expert_cache.py (_S_* constants). Keep both in sync
#define HY_S_SLOTOF       0      // int32 [Lcap * EMAX]   slot of (layer, expert) or -1
#define HY_S_OWNER        1      // int32 [Scap]          layer * EMAX + expert, or -1
#define HY_S_REFB         2      // int32 [Scap]          CLOCK reference bit
#define HY_S_PIN          3      // int32 [Scap]          epoch of the last call that used the slot; -1 = disabled
#define HY_S_SLOTBASE     4      // int64 [Scap]          device address of the slot
#define HY_S_CTL          5      // int32 [HY_CTL_SIZE]
#define HY_S_STATS        6      // int64 [HY_STATS_SIZE]
#define HY_S_GJOBS        7      // int64 [GMAX * 3]      admission copies: dst, src, bytes
#define HY_S_HIST         8      // float [Lcap * EMAX]   decayed pick counts
#define HY_S_REG_TABLES   9      // int64 [Lcap * T]      table base address (0 = unused entry)
#define HY_S_REG_TSTRIDE 10      // int32 [Lcap * T]      table stride (elements) per expert
#define HY_S_REG_TPROJ   11      // int32 [Lcap * T]      projection of the entry (0 gate, 1 up, 2 down)
#define HY_S_REG_PROJOFF 12      // int64 [Lcap * 3]      byte offset of each projection in an expert block
#define HY_S_REG_HOME    13      // int64 [Lcap * EMAX]   device address of the expert's mapped host home, 0 = not GPU-readable
#define HY_S_REG_BOUNCE  14      // int64 [Lcap]          placeholder block (valid VRAM, never computed from)
#define HY_S_REG_BYTES   15      // int64 [Lcap]          expert block bytes (trellis g|u|d)
#define HY_S_REG_CLASS   16      // int32 [Lcap]          slot class of the layer, -1 = none
#define HY_S_REG_CPULOC  17      // int32 [Lcap * NODES * EMAX] worker-local expert index per node, -1 = absent
#define HY_S_SCAP        18      // value
#define HY_S_LCAP        19      // value
#define HY_S_GMAX        20      // value
#define HY_S_COUNT       21

// ctl words
#define HY_CTL_EPOCH      0
#define HY_CTL_GCOUNT     1
#define HY_CTL_BG_ADMIT   2      // max frequency-gated background admissions per decode call (0 = off)
#define HY_CTL_HAND       8      // + class
#define HY_CTL_CLASS_LO  16      // + 2 * class
#define HY_CTL_CLASS_HI  17      // + 2 * class
#define HY_CTL_SIZE      32

// stats words
#define HY_ST_HITS        0      // unique (layer call, expert) hits
#define HY_ST_MISSES      1
#define HY_ST_ADMITS      2
#define HY_ST_EVICTS      3
#define HY_ST_NO_VICTIM   4
#define HY_ST_ZC          5      // non-resident experts computed on the GPU from the host home
#define HY_ST_CPU         6      // non-resident experts computed on a CPU worker
#define HY_ST_CALLS       7
#define HY_ST_NO_LANE     8      // picks with no lane at all (must stay 0; a configuration error)
#define HY_ST_PREFILL     9
#define HY_ST_ADMIT_BYTES 10     // bytes copied into slots by admissions (host -> VRAM over PCIe)
#define HY_ST_ZC_BYTES   11      // expert bytes of zero-copy lanes (lower bound of the PCIe reads)
#define HY_ST_BG_ADMITS  12      // admissions of CPU-lane misses (subset of HY_ST_ADMITS)
#define HY_ST_BG_REJECTS 13      // CPU-lane misses not admitted: the CLOCK victim was used as often
#define HY_ST_CPU_NODE   16      // + node: non-resident experts sent to that CPU node
#define HY_STATS_SIZE    32

// lane cost model (decode, gpu_max < 0): float CPU [4], microseconds, measured by the dispatcher
#define HY_LC_HIT         0      // GPU time per resident expert
#define HY_LC_ZC          1      // GPU time per zero-copy (or admitted) expert
#define HY_LC_CPU_FIXED   2      // CPU job round trip (issue, wake, readback) per node used
#define HY_LC_CPU         3      // CPU compute per expert of a job (its weights are streamed once)

// nodes descriptor rows (int64 CPU [HY_ND_ROWS, nodes])
#define HY_ND_SEL         0      // int32* job slot selections (pinned, mapped)
#define HY_ND_X           1      // half* job slot activations
#define HY_ND_W           2      // half* job slot weights
#define HY_ND_COUNT       3      // int32* per-slot gate array (device)
#define HY_ND_SLOT        4      // slot index into the gate array
#define HY_ND_XCAP        5      // capacity of the activation section, halves
#define HY_ND_SELCAP      6      // capacity of the selection / weight sections, entries
#define HY_ND_ROWS        7

// lanes (per expert, per call)
#define HY_LANE_NONE     -2
#define HY_LANE_GPU      -1      // resident (hit or admitted this call)
#define HY_LANE_ZC       -3      // GPU, read from the mapped host home

// host ops (moe_hybrid_host_ops)
#define HY_OP_EVICT       0      // (s): owner -> rows back to home, slot free
#define HY_OP_DISABLE     1      // (s): evict + pin = -1
#define HY_OP_ENABLE      2      // (s): pin = 0, free (slot base must be written first)
#define HY_OP_INSTALL     3      // (L, e, s): make (L, e) resident in s (data copied by the host beforehand)
#define HY_OP_RESET_LAYER 4      // (L, E): every row of L to its home (or bounce), slotof = -1

void moe_hybrid_issue
(
    const at::Tensor& state,        // int64 CPU [HY_S_COUNT]
    int64_t layer,
    int64_t num_experts,
    const at::Tensor& sel,          // int64 [rows * topk] router ids (device)
    const at::Tensor& y,            // half [rows, h] (device, contiguous)
    const at::Tensor& w,            // half [rows * topk] (device)
    at::Tensor sel_gpu,             // int64 [>= rows * topk] out: router id for GPU lanes, num_experts = masked
    const c10::optional<at::Tensor>& lane_out,   // int32 [num_experts] out (optional)
    const c10::optional<at::Tensor>& cpu_sel,    // int32 [nodes, rows * topk] out, prefill mode (optional)
    const at::Tensor& nodes,        // int64 CPU [HY_ND_ROWS, nodes] (decode mode; ignored in prefill). A node
                                    // column of all zeros is absent (no experts of this layer there)
    const at::Tensor& node_cost,    // float CPU [nodes] relative cost per assignment (adaptive policy)
    int64_t hi,                     // worker input width (zero-padded)
    int64_t prefill,                // 1: prefill mode (no admission, CPU picks into cpu_sel)
    int64_t admit,                  // 1: admission allowed for this call
    int64_t gpu_max,                // decode: misses kept on the GPU (>= 0), or -1 = derive from cpu_frac
    double cpu_frac,                // decode: share of GPU-readable misses sent to CPU lanes when gpu_max < 0
                                    // and the cost model is not measured yet
    const at::Tensor& lane_cost,    // float CPU [4] HY_LC_*: when ZC and CPU costs are > 0, decode misses
                                    // go to a CPU node only if that lowers the call's slowest lane
    int64_t prefill_gpu_min,        // prefill: min rows for a non-resident expert to run zero-copy on the GPU
    int64_t node_policy,            // 0 local/sharded, 1 balanced, 2 adaptive
    int64_t pref_node               // preferred node for replicated experts (GPU-local), -1 = none
);

void moe_hybrid_gather
(
    const at::Tensor& state,
    const at::Tensor& like,         // any tensor on the target device (stream / guard)
    int64_t blocks_per_job
);

void moe_hybrid_host_ops
(
    const at::Tensor& state,
    const at::Tensor& ops,          // int64 device [k, 4]: op, L, e, s
    int64_t num_ops
);
