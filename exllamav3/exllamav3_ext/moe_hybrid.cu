#include "moe_hybrid.cuh"
#include <cmath>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"

// Shared GPU expert cache + hybrid lane assignment. Design notes (see also moe_hybrid.cuh):
//
// - The cache owns, for every registered MoE layer, the per-expert trellis rows of that layer's
//   pointer tables (MultiLinear.ptrs_trellis of gate/up/down and the stacked gate/up table of
//   BC_BlockSparseMLP). A row points either at a VRAM slot (resident expert) or at the expert's
//   pinned + mapped host home (device alias from cudaHostGetDevicePointer, never the host VA),
//   or, for experts whose home is not GPU-readable, at a per-layer bounce block (valid VRAM that
//   is never computed from: those experts only ever take a CPU lane or a slot installed by the
//   host). suh/svh/bias stay per-expert VRAM tensors and are never touched.
//
// - The unmodified MoE kernels (coop decode, fused exl3_moe, batched reconstruct) then compute
//   whatever the tables point at, so a GPU lane is numerically identical to the all-VRAM model.
//
// - Replacement is CLOCK over the slots of the layer's size class, shared by all layers (and
//   all components) on the device. The issue kernel runs on the compute stream ahead of the
//   layer's MoE kernels; the gather kernel that fills admitted slots runs right after it. Since
//   every MoE kernel that could read a slot was enqueued earlier on the same stream, a victim
//   is never overwritten while in use; within one call, the pin epoch keeps the call's own hits
//   and admissions out of victim selection.

#define HY_THREADS 1024
#define HY_SCAN_MAX 8192
#define HY_GATHER_THREADS 256
#define HY_BG_HYST 1.25f   // background admission: candidate history must exceed the victim's by this factor

struct HyState
{
    int32_t* slotof;
    int32_t* owner;
    int32_t* refb;
    int32_t* pin;
    int64_t* slot_base;
    int32_t* ctl;
    unsigned long long* stats;
    int64_t* gjobs;
    float* hist;
    int64_t* reg_tables;
    int32_t* reg_tstride;
    int32_t* reg_tproj;
    int64_t* reg_projoff;
    int64_t* reg_home;
    int64_t* reg_bounce;
    int64_t* reg_bytes;
    int32_t* reg_class;
    int32_t* reg_cpuloc;
    int scap;
    int lcap;
    int gmax;
};

struct HyNodes
{
    int64_t h_sel[HY_MAX_NODES];
    int64_t h_x[HY_MAX_NODES];
    int64_t h_w[HY_MAX_NODES];
    int64_t dev_count[HY_MAX_NODES];
    int slot[HY_MAX_NODES];
    float cost[HY_MAX_NODES];
    int n;
};

struct HyLaneCost
{
    float hit, zc, cpu_fixed, cpu;   // HY_LC_* (microseconds); zc <= 0 or cpu <= 0 = not measured
};

static HyState hy_state(const at::Tensor& t)
{
    TORCH_CHECK(t.device().is_cpu() && t.dtype() == at::kLong && t.is_contiguous() && t.numel() >= HY_S_COUNT,
                "moe_hybrid: state descriptor must be a contiguous int64 CPU tensor of HY_S_COUNT entries");
    const int64_t* d = t.data_ptr<int64_t>();
    HyState s;
    s.slotof      = reinterpret_cast<int32_t*>(d[HY_S_SLOTOF]);
    s.owner       = reinterpret_cast<int32_t*>(d[HY_S_OWNER]);
    s.refb        = reinterpret_cast<int32_t*>(d[HY_S_REFB]);
    s.pin         = reinterpret_cast<int32_t*>(d[HY_S_PIN]);
    s.slot_base   = reinterpret_cast<int64_t*>(d[HY_S_SLOTBASE]);
    s.ctl         = reinterpret_cast<int32_t*>(d[HY_S_CTL]);
    s.stats       = reinterpret_cast<unsigned long long*>(d[HY_S_STATS]);
    s.gjobs       = reinterpret_cast<int64_t*>(d[HY_S_GJOBS]);
    s.hist        = reinterpret_cast<float*>(d[HY_S_HIST]);
    s.reg_tables  = reinterpret_cast<int64_t*>(d[HY_S_REG_TABLES]);
    s.reg_tstride = reinterpret_cast<int32_t*>(d[HY_S_REG_TSTRIDE]);
    s.reg_tproj   = reinterpret_cast<int32_t*>(d[HY_S_REG_TPROJ]);
    s.reg_projoff = reinterpret_cast<int64_t*>(d[HY_S_REG_PROJOFF]);
    s.reg_home    = reinterpret_cast<int64_t*>(d[HY_S_REG_HOME]);
    s.reg_bounce  = reinterpret_cast<int64_t*>(d[HY_S_REG_BOUNCE]);
    s.reg_bytes   = reinterpret_cast<int64_t*>(d[HY_S_REG_BYTES]);
    s.reg_class   = reinterpret_cast<int32_t*>(d[HY_S_REG_CLASS]);
    s.reg_cpuloc  = reinterpret_cast<int32_t*>(d[HY_S_REG_CPULOC]);
    s.scap = (int) d[HY_S_SCAP];
    s.lcap = (int) d[HY_S_LCAP];
    s.gmax = (int) d[HY_S_GMAX];
    TORCH_CHECK(s.slotof && s.owner && s.refb && s.pin && s.slot_base && s.ctl && s.stats && s.gjobs &&
                s.hist && s.reg_tables && s.reg_tstride && s.reg_tproj && s.reg_projoff && s.reg_home &&
                s.reg_bounce && s.reg_bytes && s.reg_class && s.reg_cpuloc,
                "moe_hybrid: state descriptor has a null array");
    return s;
}

// ---- device helpers ------------------------------------------------------------------------

__device__ __forceinline__ int64_t hy_home(const HyState& st, int L, int e)
{
    const int64_t h = st.reg_home[(int64_t) L * HY_EMAX + e];
    return h ? h : st.reg_bounce[L];
}

// Point every registered table row of (L, e) at the expert block starting at `base`
__device__ void hy_repoint(const HyState& st, int L, int e, int64_t base)
{
    for (int t = 0; t < HY_MAX_TABLES; ++t)
    {
        const int r = L * HY_MAX_TABLES + t;
        const int64_t tab = st.reg_tables[r];
        if (!tab) continue;
        const int p = st.reg_tproj[r];
        int64_t* row = reinterpret_cast<int64_t*>(tab) + (int64_t) e * st.reg_tstride[r];
        *row = base + st.reg_projoff[L * 3 + p];
    }
}

// Release slot s: its tenant's rows go back to the tenant's home (zero-copy) or bounce block
__device__ void hy_evict(const HyState& st, int s)
{
    const int o = st.owner[s];
    if (o < 0) return;
    const int L2 = o / HY_EMAX;
    const int e2 = o % HY_EMAX;
    st.slotof[o] = -1;
    hy_repoint(st, L2, e2, hy_home(st, L2, e2));
    st.owner[s] = -1;
    st.stats[HY_ST_EVICTS] += 1;
}

__device__ void hy_install(const HyState& st, int L, int e, int s)
{
    hy_evict(st, s);
    const int key = L * HY_EMAX + e;
    st.owner[s] = key;
    st.slotof[key] = s;
    hy_repoint(st, L, e, st.slot_base[s]);
}

// CLOCK victim in the class range [lo, lo + range), advancing *hand; skips disabled slots and
// slots this call already uses (pin == epoch), clears reference bits on the way. -1 if none
__device__ int hy_clock_victim(const HyState& st, int lo, int range, int* hand, int epoch)
{
    const int steps = min(2 * range, HY_SCAN_MAX);
    for (int k = 0; k < steps; ++k)
    {
        const int s = lo + *hand;
        *hand = *hand + 1 == range ? 0 : *hand + 1;
        const int p = st.pin[s];
        if (p < 0 || p == epoch) continue;
        if (st.refb[s]) { st.refb[s] = 0; continue; }
        return s;
    }
    return -1;
}

__device__ __forceinline__ int hy_cpuloc(const HyState& st, int L, int node, int e)
{
    return st.reg_cpuloc[((int64_t) L * HY_MAX_NODES + node) * HY_EMAX + e];
}

// CPU node for expert e among the nodes holding a worker copy, by policy; -1 if none holds it
__device__ int hy_pick_node(const HyState& st, int L, int e, int cnt, int nn, int policy, int pref,
                            const int* load, const float* cost)
{
    int best = -1;
    float best_c = 0.0f;
    for (int nd = 0; nd < nn; ++nd)
    {
        if (hy_cpuloc(st, L, nd, e) < 0) continue;
        if (policy == 0)
        {
            if (nd == pref) return nd;
            if (best < 0) best = nd;
            continue;
        }
        // Explicit round-to-nearest multiply (no contraction), so the reference reproduces it
        const float c = policy == 1 ? (float) load[nd] : __fmul_rn((float) (load[nd] + cnt), cost[nd]);
        if (best < 0 || c < best_c) { best = nd; best_c = c; }
    }
    return best;
}

// ---- issue kernel ----------------------------------------------------------------------------

__global__ __launch_bounds__(HY_THREADS)
void moe_hybrid_issue_kernel
(
    const HyState st,
    const HyNodes nd,
    const int L,
    const int E,
    const int64_t* __restrict__ sel,
    const int n,
    const half* __restrict__ y,
    const int h_,
    const int hi,
    const half* __restrict__ w,
    int64_t* __restrict__ sel_gpu,
    int32_t* __restrict__ lane_out,
    int32_t* __restrict__ cpu_sel,
    const int prefill,
    const int admit,
    const int gpu_max,
    const float cpu_frac,
    const HyLaneCost lc,
    const int prefill_gpu_min,
    const int node_policy,
    const int pref_node
)
{
    __shared__ int s_cnt[HY_EMAX];
    __shared__ int s_lane[HY_EMAX];
    __shared__ int s_miss[HY_EMAX];
    __shared__ float s_key[HY_EMAX];
    __shared__ int s_nmiss;
    __shared__ int s_hits;
    __shared__ int s_epoch;
    __shared__ int s_any[HY_MAX_NODES];

    const int t = threadIdx.x;
    const int64_t hbase = (int64_t) L * HY_EMAX;

    for (int e = t; e < HY_EMAX; e += blockDim.x)
    {
        s_cnt[e] = 0;
        s_lane[e] = HY_LANE_NONE;
    }
    if (t < HY_MAX_NODES) s_any[t] = 0;
    if (t == 0)
    {
        s_nmiss = 0;
        s_hits = 0;
        // Epoch stays positive (pin -1 = disabled, 0 = never used): wraps within 30 bits
        int ep = (st.ctl[HY_CTL_EPOCH] + 1) & 0x3fffffff;
        if (ep == 0) ep = 1;
        st.ctl[HY_CTL_EPOCH] = ep;
        st.ctl[HY_CTL_GCOUNT] = 0;
        s_epoch = ep;
    }
    __syncthreads();

    // Tokens per expert
    for (int i = t; i < n; i += blockDim.x)
    {
        const int64_t e = sel[i];
        if (e >= 0 && e < E) atomicAdd(&s_cnt[e], 1);
    }
    __syncthreads();

    // Hits (parallel over experts); misses collected for the serial lane pass
    const int epoch = s_epoch;
    for (int e = t; e < E; e += blockDim.x)
    {
        const int c = s_cnt[e];
        if (!c) continue;
        st.hist[hbase + e] += (float) c;
        const int s = st.slotof[hbase + e];
        if (s >= 0)
        {
            st.refb[s] = 1;
            st.pin[s] = epoch;
            s_lane[e] = HY_LANE_GPU;
            atomicAdd(&s_hits, 1);
        }
        else
        {
            const int k = atomicAdd(&s_nmiss, 1);
            s_miss[k] = e;
            s_key[k] = st.hist[hbase + e];
        }
    }
    __syncthreads();

    if (t == 0)
    {
        const int m = s_nmiss;
        st.stats[HY_ST_CALLS] += 1;
        st.stats[HY_ST_HITS] += s_hits;
        st.stats[HY_ST_MISSES] += m;
        if (prefill) st.stats[HY_ST_PREFILL] += 1;

        // Deterministic order (the parallel collection above is not): decode orders the misses
        // hottest first (more tokens in this call, then more recent picks), so the GPU quota and
        // the admissions go to the experts most likely to be reused; prefill decides per expert
        // by its own row count and only needs the order for reproducibility
        for (int i = 1; i < m; ++i)
        {
            const int e = s_miss[i];
            const float k = s_key[i];
            int j = i - 1;
            while (j >= 0)
            {
                const int ej = s_miss[j];
                bool after;
                if (prefill) after = ej > e;
                else
                {
                    const int ci = s_cnt[e], cj = s_cnt[ej];
                    after = cj < ci || (cj == ci && (s_key[j] < k || (s_key[j] == k && ej > e)));
                }
                if (!after) break;
                s_miss[j + 1] = ej;
                s_key[j + 1] = s_key[j];
                --j;
            }
            s_miss[j + 1] = e;
            s_key[j + 1] = k;
        }

        // Flexible misses: GPU-readable home and a CPU copy, so either lane can take them.
        // Forced GPU: GPU-readable home, no CPU copy
        int nflex = 0;
        int nforced_gpu = 0;
        for (int j = 0; j < m; ++j)
        {
            const int e = s_miss[j];
            if (!st.reg_home[hbase + e]) continue;
            bool has_cpu = false;
            for (int q = 0; q < nd.n && !has_cpu; ++q) has_cpu = hy_cpuloc(st, L, q, e) >= 0;
            nflex += has_cpu;
            nforced_gpu += !has_cpu;
        }
        // Decode lane split for the flexible misses:
        // - gpu_max >= 0: the first gpu_max (hottest first) stay on the GPU
        // - cost model measured: greedy min-max. Each flexible miss goes to its CPU node only if
        //   that makes the call's slowest lane (GPU work vs every CPU node's round trip plus
        //   compute) strictly shorter than keeping it on the GPU; ties stay on the GPU. So the
        //   CPU never lengthens the layer's critical path under the measured costs
        // - otherwise: share cpu_frac of the flexible misses to the CPU (bootstrap until measured)
        const bool use_cost = gpu_max < 0 && lc.zc > 0.0f && lc.cpu > 0.0f;
        int quota = gpu_max >= 0 ? min(gpu_max, nflex) : nflex - (int) rintf((float) nflex * cpu_frac);
        if (quota < 0) quota = 0;

        int load[HY_MAX_NODES];
        float t_node[HY_MAX_NODES];
        for (int q = 0; q < HY_MAX_NODES; ++q) { load[q] = 0; t_node[q] = 0.0f; }
        float t_gpu = __fadd_rn(__fmul_rn((float) s_hits, lc.hit), __fmul_rn((float) nforced_gpu, lc.zc));
        float t_cpu_max = 0.0f;

        const int cls = st.reg_class[L];
        const int lo = cls >= 0 ? st.ctl[HY_CTL_CLASS_LO + 2 * cls] : 0;
        const int range = cls >= 0 ? st.ctl[HY_CTL_CLASS_HI + 2 * cls] - lo : 0;
        int hand = cls >= 0 ? st.ctl[HY_CTL_HAND + cls] : 0;
        if (range > 0 && (hand < 0 || hand >= range)) hand = 0;
        int gcount = 0;
        int bg = 0;
        const int bg_max = st.ctl[HY_CTL_BG_ADMIT];
        const int gpu_gate = st.ctl[HY_CTL_GPU_GATE];
        int flex_gpu = 0;
        const int64_t ebytes = st.reg_bytes[L];

        // Two passes in miss order: experts with one possible lane first (their load is fixed),
        // then the flexible ones, decided against that load
        for (int pass = 0; pass < 2; ++pass)
        for (int j = 0; j < m; ++j)
        {
            const int e = s_miss[j];
            const int c = s_cnt[e];
            const bool mapped = st.reg_home[hbase + e] != 0;
            bool has_cpu = false;
            for (int q = 0; q < nd.n && !has_cpu; ++q) has_cpu = hy_cpuloc(st, L, q, e) >= 0;
            const bool flex = mapped && has_cpu;
            if ((int) flex != pass) continue;
            const int node = hy_pick_node(st, L, e, c, nd.n, node_policy, pref_node, load, nd.cost);

            bool to_gpu;
            if (!mapped) to_gpu = false;
            else if (node < 0) to_gpu = true;
            else if (prefill) to_gpu = c >= prefill_gpu_min;
            else if (use_cost)
            {
                const float add_cpu = __fadd_rn(t_node[node] > 0.0f ? 0.0f : lc.cpu_fixed, lc.cpu);
                const float m_gpu = fmaxf(__fadd_rn(t_gpu, lc.zc), t_cpu_max);
                const float m_cpu = fmaxf(t_gpu, fmaxf(t_cpu_max, __fadd_rn(t_node[node], add_cpu)));
                to_gpu = !(m_cpu < m_gpu);
            }
            else
            {
                to_gpu = flex_gpu < quota;
                flex_gpu += to_gpu;
            }

            int lane = HY_LANE_NONE;
            if (to_gpu)
            {
                lane = HY_LANE_ZC;
                if (pass == 1) t_gpu = __fadd_rn(t_gpu, lc.zc);
                if (admit && !prefill && range > 0 && gcount < st.gmax)
                {
                    // With the gate on, a slot is only taken from an expert picked at most as often
                    // than e (decayed history); otherwise e runs zero-copy this call, which reads
                    // the same bytes over PCIe without evicting a hotter expert
                    const int victim = hy_clock_victim(st, lo, range, &hand, epoch);
                    const int o = victim >= 0 ? st.owner[victim] : -1;
                    if (victim >= 0 && gpu_gate && o >= 0 && st.hist[hbase + e] < st.hist[o])
                        st.stats[HY_ST_GPU_REJECTS] += 1;
                    else if (victim >= 0)
                    {
                        hy_install(st, L, e, victim);
                        st.refb[victim] = 1;
                        st.pin[victim] = epoch;
                        int64_t* job = st.gjobs + 3 * gcount;
                        job[0] = st.slot_base[victim];
                        job[1] = st.reg_home[hbase + e];
                        job[2] = ebytes;
                        gcount++;
                        lane = HY_LANE_GPU;
                        st.stats[HY_ST_ADMITS] += 1;
                        st.stats[HY_ST_ADMIT_BYTES] += (unsigned long long) ebytes;
                    }
                    else st.stats[HY_ST_NO_VICTIM] += 1;
                }
                if (lane == HY_LANE_ZC)
                {
                    st.stats[HY_ST_ZC] += 1;
                    st.stats[HY_ST_ZC_BYTES] += (unsigned long long) ebytes;
                }
            }
            else if (node >= 0)
            {
                lane = node;
                load[node] += c;
                t_node[node] = __fadd_rn(t_node[node], __fadd_rn(t_node[node] > 0.0f ? 0.0f : lc.cpu_fixed, lc.cpu));
                t_cpu_max = fmaxf(t_cpu_max, t_node[node]);
                st.stats[HY_ST_CPU] += 1;
                st.stats[HY_ST_CPU_NODE + node] += 1;

                // Background admission: the CPU computes this call's picks of e either way, but
                // when e is used clearly more often than the slot's CLOCK victim, copy it into
                // the slot for the next calls. Without this, once the cost model routes most
                // misses to the CPU the cache only refreshes from the few GPU-lane misses and
                // goes stale. The copy is gated by frequency (decayed pick history) so a
                // one-off expert never displaces a hot one, and bounded per call
                if (admit && !prefill && mapped && range > 0 && gcount < st.gmax && bg < bg_max)
                {
                    const int victim = hy_clock_victim(st, lo, range, &hand, epoch);
                    if (victim >= 0)
                    {
                        const int o = st.owner[victim];
                        const float hv = o >= 0 ? st.hist[o] : 0.0f;
                        if (o < 0 || st.hist[hbase + e] > __fmul_rn(hv, HY_BG_HYST))
                        {
                            hy_install(st, L, e, victim);
                            st.refb[victim] = 1;
                            st.pin[victim] = epoch;
                            int64_t* job = st.gjobs + 3 * gcount;
                            job[0] = st.slot_base[victim];
                            job[1] = st.reg_home[hbase + e];
                            job[2] = ebytes;
                            gcount++;
                            bg++;
                            st.stats[HY_ST_ADMITS] += 1;
                            st.stats[HY_ST_BG_ADMITS] += 1;
                            st.stats[HY_ST_ADMIT_BYTES] += (unsigned long long) ebytes;
                        }
                        else st.stats[HY_ST_BG_REJECTS] += 1;
                    }
                    else st.stats[HY_ST_NO_VICTIM] += 1;
                }
            }
            else st.stats[HY_ST_NO_LANE] += 1;
            s_lane[e] = lane;
        }
        if (cls >= 0) st.ctl[HY_CTL_HAND + cls] = hand;
        st.ctl[HY_CTL_GCOUNT] = gcount;
    }
    __syncthreads();

    // GPU selection: lanes computed on the GPU keep their router id (the table row), everything
    // else is masked with an id the kernels treat as out of range
    for (int i = t; i < n; i += blockDim.x)
    {
        const int64_t e = sel[i];
        const int lane = (e >= 0 && e < E) ? s_lane[e] : HY_LANE_NONE;
        sel_gpu[i] = (lane == HY_LANE_GPU || lane == HY_LANE_ZC) ? e : (int64_t) E;
    }
    if (lane_out)
        for (int e = t; e < E; e += blockDim.x) lane_out[e] = s_lane[e];

    // CPU lanes
    for (int q = 0; q < nd.n; ++q)
    {
        int32_t* hs = prefill ? (cpu_sel ? cpu_sel + (int64_t) q * n : nullptr)
                              : reinterpret_cast<int32_t*>(nd.h_sel[q]);
        if (!hs) continue;
        int any = 0;
        for (int i = t; i < n; i += blockDim.x)
        {
            const int64_t e = sel[i];
            const int lane = (e >= 0 && e < E) ? s_lane[e] : HY_LANE_NONE;
            const int32_t v = lane == q ? hy_cpuloc(st, L, q, (int) e) : -1;
            hs[i] = v;
            any |= v >= 0;
        }
        if (any) atomicOr(&s_any[q], 1);
    }
    if (prefill) return;
    __syncthreads();

    // Decode: routing weights straight into each active node's pinned slot (zero-copy stores,
    // as moe_split_issue) and the per-slot gate the collect kernel reads. The activations are
    // staged by moe_hybrid_stage_x_kernel, launched right after on the same stream
    for (int q = 0; q < nd.n; ++q)
    {
        if (!nd.dev_count[q]) continue;     // absent node (no experts of this layer)
        const int active = s_any[q];
        if (t == 0) reinterpret_cast<int32_t*>(nd.dev_count[q])[nd.slot[q]] = active;
        if (!active) continue;
        half* hw = reinterpret_cast<half*>(nd.h_w[q]);
        for (int i = t; i < n; i += blockDim.x) hw[i] = w[i];
    }
    __threadfence_system();
}

// Activations of the decode-size job into every node slot whose gate the issue kernel set,
// zero-padded to the worker's input width (vectorized when both widths are 8-half aligned;
// PCIe write-combining wants the widest coalesced stores). Separate from the issue kernel so
// the copy runs on many blocks rather than the issue kernel's single one
__global__ void moe_hybrid_stage_x_kernel
(
    const HyNodes nd,
    const half* __restrict__ y,
    const int rows,
    const int h_,
    const int hi
)
{
    for (int q = 0; q < nd.n; ++q)
    {
        if (!nd.dev_count[q] || !reinterpret_cast<const int32_t*>(nd.dev_count[q])[nd.slot[q]]) continue;
        half* hx = reinterpret_cast<half*>(nd.h_x[q]);
        const int total = rows * hi;
        if ((h_ & 7) == 0 && (hi & 7) == 0)
        {
            const int h8 = h_ >> 3, hi8 = hi >> 3;
            const int4* y4 = reinterpret_cast<const int4*>(y);
            int4* x4 = reinterpret_cast<int4*>(hx);
            const int4 z = {0, 0, 0, 0};
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * hi8; i += gridDim.x * blockDim.x)
            {
                const int row = i / hi8;
                const int c = i - row * hi8;
                x4[i] = c < h8 ? y4[row * h8 + c] : z;
            }
        }
        else
        {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x)
            {
                const int row = i / hi;
                const int c = i - row * hi;
                hx[i] = c < h_ ? y[row * h_ + c] : __float2half_rn(0.0f);
            }
        }
    }
    __threadfence_system();
}

void moe_hybrid_issue
(
    const at::Tensor& state,
    int64_t layer,
    int64_t num_experts,
    const at::Tensor& sel,
    const at::Tensor& y,
    const at::Tensor& w,
    at::Tensor sel_gpu,
    const c10::optional<at::Tensor>& lane_out,
    const c10::optional<at::Tensor>& cpu_sel,
    const at::Tensor& nodes,
    const at::Tensor& node_cost,
    int64_t hi,
    int64_t prefill,
    int64_t admit,
    int64_t gpu_max,
    double cpu_frac,
    const at::Tensor& lane_cost,
    int64_t prefill_gpu_min,
    int64_t node_policy,
    int64_t pref_node
)
{
    const at::cuda::OptionalCUDAGuard device_guard(sel.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const HyState st = hy_state(state);

    TORCH_CHECK_DTYPE(sel, kLong);
    TORCH_CHECK_DTYPE(sel_gpu, kLong);
    TORCH_CHECK_DTYPE(y, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE_OPT(lane_out, kInt);
    TORCH_CHECK_DTYPE_OPT(cpu_sel, kInt);
    TORCH_CHECK(sel.is_contiguous() && y.is_contiguous() && w.is_contiguous() && sel_gpu.is_contiguous(),
                "moe_hybrid_issue: inputs must be contiguous");
    TORCH_CHECK(sel.device() == y.device() && sel.device() == w.device() && sel.device() == sel_gpu.device(),
                "moe_hybrid_issue: device mismatch");
    TORCH_CHECK(y.dim() == 2, "moe_hybrid_issue: y must be (rows, h)");
    TORCH_CHECK(layer >= 0 && layer < st.lcap, "moe_hybrid_issue: layer index out of range");
    TORCH_CHECK(num_experts > 0 && num_experts <= HY_EMAX, "moe_hybrid_issue: too many experts");
    const int n = (int) sel.numel();
    const int rows = (int) y.size(0);
    const int h_ = (int) y.size(1);
    TORCH_CHECK(w.numel() == n && (rows ? n % rows == 0 : n == 0), "moe_hybrid_issue: sel and w must hold rows * topk entries");
    TORCH_CHECK(sel_gpu.numel() >= n, "moe_hybrid_issue: sel_gpu too small");
    TORCH_CHECK(hi >= h_, "moe_hybrid_issue: worker width below the row width");
    if (lane_out) TORCH_CHECK(lane_out.value().numel() >= num_experts && lane_out.value().is_contiguous(), "moe_hybrid_issue: lane_out too small");

    TORCH_CHECK(nodes.device().is_cpu() && nodes.dtype() == at::kLong && nodes.is_contiguous() && nodes.dim() == 2 && nodes.size(0) == HY_ND_ROWS,
                "moe_hybrid_issue: nodes must be an int64 CPU tensor (HY_ND_ROWS, num_nodes)");
    TORCH_CHECK(lane_cost.device().is_cpu() && lane_cost.dtype() == at::kFloat && lane_cost.is_contiguous() && lane_cost.numel() >= 4,
                "moe_hybrid_issue: lane_cost must be a float CPU tensor of 4 entries");
    HyLaneCost lc;
    {
        const float* l = lane_cost.data_ptr<float>();
        lc.hit = l[HY_LC_HIT]; lc.zc = l[HY_LC_ZC]; lc.cpu_fixed = l[HY_LC_CPU_FIXED]; lc.cpu = l[HY_LC_CPU];
        TORCH_CHECK(std::isfinite(lc.hit) && std::isfinite(lc.zc) && std::isfinite(lc.cpu_fixed) && std::isfinite(lc.cpu) &&
                    lc.hit >= 0.0f && lc.zc >= 0.0f && lc.cpu_fixed >= 0.0f && lc.cpu >= 0.0f,
                    "moe_hybrid_issue: lane costs must be finite and non-negative");
    }
    TORCH_CHECK(node_cost.device().is_cpu() && node_cost.dtype() == at::kFloat && node_cost.numel() >= nodes.size(1),
                "moe_hybrid_issue: node_cost must be a float CPU tensor (num_nodes)");
    HyNodes nd {};
    nd.n = (int) nodes.size(1);
    TORCH_CHECK(nd.n >= 0 && nd.n <= HY_MAX_NODES, "moe_hybrid_issue: too many nodes");
    const int64_t* ndp = nodes.data_ptr<int64_t>();
    const float* costp = node_cost.data_ptr<float>();
    for (int q = 0; q < nd.n; ++q)
    {
        nd.h_sel[q] = ndp[HY_ND_SEL * nd.n + q];
        nd.h_x[q] = ndp[HY_ND_X * nd.n + q];
        nd.h_w[q] = ndp[HY_ND_W * nd.n + q];
        nd.dev_count[q] = ndp[HY_ND_COUNT * nd.n + q];
        nd.slot[q] = (int) ndp[HY_ND_SLOT * nd.n + q];
        nd.cost[q] = costp[q] > 0.0f ? costp[q] : 1.0f;
        if (!prefill)
        {
            const bool any = nd.h_sel[q] || nd.h_x[q] || nd.h_w[q] || nd.dev_count[q];
            const bool all = nd.h_sel[q] && nd.h_x[q] && nd.h_w[q] && nd.dev_count[q];
            TORCH_CHECK(any == all, "moe_hybrid_issue: a decode node column must be complete or all zero");
            // The kernels write rows * hi activations and rows * topk selections / weights into
            // the slot: a call larger than the slot would corrupt the pinned job ring
            if (all)
                TORCH_CHECK((int64_t) rows * hi <= ndp[HY_ND_XCAP * nd.n + q] && (int64_t) n <= ndp[HY_ND_SELCAP * nd.n + q],
                            "moe_hybrid_issue: decode job exceeds the CPU slot capacity");
        }
    }
    if (prefill && cpu_sel)
    {
        TORCH_CHECK(cpu_sel.value().is_contiguous() && cpu_sel.value().numel() >= (int64_t) nd.n * n,
                    "moe_hybrid_issue: cpu_sel must hold num_nodes * rows * topk entries");
        TORCH_CHECK(cpu_sel.value().device() == sel.device(), "moe_hybrid_issue: cpu_sel device mismatch");
    }

    // n = 0 still launches: decode slots must read an empty gate, the epoch must advance
    moe_hybrid_issue_kernel<<<1, HY_THREADS, 0, stream>>>
    (
        st, nd, (int) layer, (int) num_experts,
        (const int64_t*) sel.data_ptr(), n,
        (const half*) y.data_ptr(), h_, (int) hi,
        (const half*) w.data_ptr(),
        (int64_t*) sel_gpu.data_ptr(),
        lane_out ? (int32_t*) lane_out.value().data_ptr() : nullptr,
        (prefill && cpu_sel) ? (int32_t*) cpu_sel.value().data_ptr() : nullptr,
        (int) prefill, (int) admit, (int) gpu_max, (float) cpu_frac, lc, (int) prefill_gpu_min,
        (int) node_policy, (int) pref_node
    );
    cuda_check(cudaPeekAtLastError());

    if (!prefill && nd.n && rows)
    {
        const int total = rows * (int) hi;
        int threads = 256;
        int blocks = (total / 8 + threads - 1) / threads;
        if (blocks < 1) blocks = 1;
        if (blocks > 64) blocks = 64;
        moe_hybrid_stage_x_kernel<<<blocks, threads, 0, stream>>>(nd, (const half*) y.data_ptr(), rows, h_, (int) hi);
        cuda_check(cudaPeekAtLastError());
    }
}

// ---- gather (admission copies) -----------------------------------------------------------

__global__ __launch_bounds__(HY_GATHER_THREADS)
void moe_hybrid_gather_kernel(const int32_t* __restrict__ ctl, const int64_t* __restrict__ gjobs)
{
    const int job = blockIdx.y;
    if (job >= ctl[HY_CTL_GCOUNT]) return;
    int4* dst = reinterpret_cast<int4*>(gjobs[3 * job + 0]);
    const int4* src = reinterpret_cast<const int4*>(gjobs[3 * job + 1]);
    const int64_t n16 = gjobs[3 * job + 2] >> 4;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = src[i];
}

void moe_hybrid_gather
(
    const at::Tensor& state,
    const at::Tensor& like,
    int64_t blocks_per_job
)
{
    const at::cuda::OptionalCUDAGuard device_guard(like.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const HyState st = hy_state(state);
    if (st.gmax <= 0) return;
    const int bx = (int) (blocks_per_job > 0 ? blocks_per_job : 16);
    dim3 grid(bx, st.gmax);
    moe_hybrid_gather_kernel<<<grid, HY_GATHER_THREADS, 0, stream>>>(st.ctl, st.gjobs);
    cuda_check(cudaPeekAtLastError());
}

// ---- host-initiated state edits ------------------------------------------------------------

__global__ void moe_hybrid_host_ops_kernel(const HyState st, const int64_t* __restrict__ ops, const int num_ops)
{
    if (threadIdx.x || blockIdx.x) return;
    for (int i = 0; i < num_ops; ++i)
    {
        const int op = (int) ops[4 * i + 0];
        const int L = (int) ops[4 * i + 1];
        const int e = (int) ops[4 * i + 2];
        const int s = (int) ops[4 * i + 3];
        switch (op)
        {
            case HY_OP_EVICT:
                if (s >= 0 && s < st.scap) hy_evict(st, s);
                break;
            case HY_OP_DISABLE:
                if (s >= 0 && s < st.scap)
                {
                    hy_evict(st, s);
                    st.pin[s] = -1;
                    st.refb[s] = 0;
                }
                break;
            case HY_OP_ENABLE:
                if (s >= 0 && s < st.scap)
                {
                    st.owner[s] = -1;
                    st.refb[s] = 0;
                    st.pin[s] = 0;
                }
                break;
            case HY_OP_INSTALL:
                if (s >= 0 && s < st.scap && L >= 0 && L < st.lcap && e >= 0 && e < HY_EMAX &&
                    st.pin[s] >= 0 && st.slotof[(int64_t) L * HY_EMAX + e] < 0)
                {
                    hy_install(st, L, e, s);
                    st.refb[s] = 0;
                }
                break;
            case HY_OP_RESET_LAYER:
                if (L >= 0 && L < st.lcap)
                {
                    // e carries the layer's expert count here
                    for (int x = 0; x < e && x < HY_EMAX; ++x)
                    {
                        const int key = L * HY_EMAX + x;
                        const int s2 = st.slotof[key];
                        if (s2 >= 0) hy_evict(st, s2);
                        else hy_repoint(st, L, x, hy_home(st, L, x));
                    }
                }
                break;
            default:
                break;
        }
    }
}

void moe_hybrid_host_ops
(
    const at::Tensor& state,
    const at::Tensor& ops,
    int64_t num_ops
)
{
    const at::cuda::OptionalCUDAGuard device_guard(ops.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const HyState st = hy_state(state);
    TORCH_CHECK_DTYPE(ops, kLong);
    TORCH_CHECK(ops.is_contiguous() && ops.numel() >= 4 * num_ops, "moe_hybrid_host_ops: ops must be (k, 4) int64");
    if (num_ops <= 0) return;
    moe_hybrid_host_ops_kernel<<<1, 32, 0, stream>>>(st, (const int64_t*) ops.data_ptr(), (int) num_ops);
    cuda_check(cudaPeekAtLastError());
}
