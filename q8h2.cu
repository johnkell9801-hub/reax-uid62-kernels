// Q8H2 (rgh, 25 Oct 2026; S1ENG_Q8H2, q8h2.py): persistent, byte-identical twins of the in-family Q8 kernels e41
// launches at H = 0 (KH = 0, Kl = K = 2560: pure int8, per-row activation scale x per-row weight scale, WTL weights):
//   q8h2_h_<small>   = q8h_b128x256_<small>_k6 (ruj's Q8K RING|WTL twin of rgh's q8h: bf16 out [M, N]; qkv|z, attn
//                      q, k|v)
//   q8h2_gr_<small>  = q8gr_b128x256_<small>_k7 (the Q8K LUT|RING|WTL twin of rgh's q8gr: gate|up + SwiGLU + ROT ->
//                      the down family's int8 hidden hq [M, N / 2] + hs [(N / 2) / 128][M])
// <small> = the tail tile of rows [mb, M) as q8.py picks it (s64 / s32 / s96 x256); tile shapes, warp layout, tile
// order (rgh's hetero order: 128-row tiles m fastest, then the tail tiles) are the twins'.
// What changes (data movement, scheduling and how many instructions an epilogue takes; never a value):
//   * PERSISTENT CTAs: grid = min(tiles, SMs); CTA b runs tile b, then the tiles an atomic counter hands out (ctr[0],
//     taken at k-tile KT - 3 so tiles go out in about the order CTAs finish; ctr[1] counts finished CTAs and the last
//     one re-zeroes both, so a captured launch replays). One launch per GEMM, no per-tile CTA launch.
//   * stages [B0 32 KB | A0 16 KB | A1 16 KB | B1 32 KB]: the NEXT tile's k-tile 0 goes into B0 | A0 during the
//     current tile's last k-tile and lands under its epilogue (which uses A1 | B1 only); the tile's sa / sw / bias are
//     cp.async'ed into smem with k-tile 1 (2.5 KB), so the dequant reads no global memory.
//   * a KH = 0, K = 2560 mainloop (no protected pass: RING at KH = 0 is this loop): base pointers + compile-time WTL
//     chunk offsets, the special k-tiles (scales, counter) peeled out of the plain loop (193 instructions per k-tile).
//   * q8gr: the SwiGLU outputs staged as their bf16 values (32 KB, swizzled; bf16 roundings two at a time with
//     __floats2bfloat162_rn = two __float2bfloat16_rn); the rotation 4 lanes per row with 32 consecutive columns per
//     lane: rgh's butterflies bits 0 .. 4 in registers, bits 5, 6 across lanes (sm_120 shuffles cost 2 cycles per warp
//     instruction per SM: 4x fewer than one warp per row with 4 columns per lane).
//   * q8h: the bf16 tile staged in A1 | B1 (swizzled 512 B rows, the 128-row tile in two 64-row phases) and stored
//     with 16-byte row stores (store_tile's coalescing; the padded 66 KB tile does not fit beside the prefetch).
// Every output element is computed by the same per-element code as the twin it replaces: the int32 sums are exact in
// any order; q8::dequant's fmaf(acc, sa sw, bias); bf16 rounding of q8h's outputs with q8::pack_bf2; q8k epi_g_rotK's
// LUT SiLU line (q8k.cubin's q8k_silu, passed by pointer); the centring (x - mu) is (__fsub_rn / __fmul_rn: never
// contracted) with the rotation sign folded into is (round(v (-s)) = -round(v s)); q8::fwht_reg and the lane
// butterflies in rgh's stage order (ascending index bits, low' = low + high, high' = low - high; across lanes one exact
// FFMA, fma(v, +-1, o)); the row amax (an order-free max); 127 / m, hs = m / 127 * ROT_RS, rn(y 127 / m).
// q8h2.py's start-up gate checks every kind byte for byte against today's kernel before the lanes use it.
// Credit: rgh (Q8 / ROT: every per-element line reused here), ruj (Q8K twins: the LUT, RING, S96, WTL wiring), pjs (the
// WTL layout).
#include "q8.cu"

#ifdef Q8H2_TM
extern "C" __device__ long long tm_acc[4096][4];   // measurement build only
extern "C" __device__ long long tm_last[4096];
__device__ long long tm_acc[4096][4];
__device__ long long tm_last[4096];
#endif

namespace q8h2 {
using namespace q8;

constexpr int BKU = 64;                       // one k-tile = 64 2-byte units = 128 int8 columns
constexpr int A_BYTES = 128 * BKU * 2;        // 16 KB: the A rows of a 128-row tile (smaller tiles use its first rows)
constexpr int B_BYTES = 256 * BKU * 2;        // 32 KB
constexpr int PIPE = 2 * (A_BYTES + B_BYTES); // 96 KB: [B0 32 KB | A0 16 KB | A1 16 KB | B1 32 KB]
constexpr int EPI = B_BYTES + A_BYTES;        // the epilogue's smem: A1 | B1 (48 KB; B0 | A0 take the next k-tile 0)
constexpr int SCL = PIPE;                     // the tile's scales: sa [128] | sw [256] | bias [256] (fp32, 2.5 KB)
constexpr int SLOT = SCL + 2560;              // the next-tile slots [2]
constexpr int SMEM_BYTES = SLOT + 16;         // 100,880 B (q8h2.py SMEM; sm_120 opt-in limit 101,376)
static_assert(SMEM_BYTES <= 101376, "smem");
// q8gr staging: 256 B rows; 16-byte chunk c of row r stored at chunk tsw(c, r): conflict-free 4-byte writes from the
// MMA layout (8 rows x 4 lanes of one chunk) and 16-byte reads of 2 rows x 4 lanes (chunks 4 q + k, q = 0 .. 3)
DEVI int tsw(int c, int r) { return c ^ (((c >> 3) & 1) << 1) ^ (r & 7); }

struct ArgsP {
    const int8_t* A;        // [M, K] int8 (lda = K)
    const int8_t* B;        // [N, K] int8, WTL order
    const float* sa;        // [M]
    const float* sw;        // [N]
    const float* bias;      // [N] or null
    void* out;              // q8h: bf16 [M, ldo]; q8gr: int8 hq [M, N / 2]
    float* hs;              // q8gr: [(N / 2) / 128][M]
    const float* mu;        // q8gr: [N / 2] or null
    const float* is;        // q8gr: [N / 2] or null
    const uint16_t* lut;    // q8gr: q8k_silu (q8k.cubin)
    int* ctr;               // [2] zero at launch (left zero by the last CTA)
    int M, N, K, ldo, mb, nt;   // nt = work items in all (tiles + ls)
    int ls;                 // TS: the last ls 128-row tiles run as two 64-row halves, after the tail tiles
};

DEVI uint32_t st_a(uint32_t s0, int s) { return s0 + B_BYTES + s * A_BYTES; }
DEVI uint32_t st_b(uint32_t s0, int s) { return s0 + s * (B_BYTES + 2 * A_BYTES); }

// the loader of a KH = 0 tile on WTL weights (q8k::LoaderK with LV = 4: the same 16-byte chunks into the same swizzled
// smem slots), addressed by base pointers + compile-time chunk offsets (K = 64 KT units): A chunk q = row r0 + 32 q of
// the tile (FULL: every row < M, no predicate; else rows >= M zero-filled from a safe address, as rgh's Loader);
// B chunk q = 32-row WTL band q of the tile's B rows (q8gr: gate / up bands interleaved per warp column)
template <class C, int KT, bool FULL>
struct LoaderW {
    static constexpr int RS = C::THREADS / C::CH;        // 32 rows per chunk row
    static constexpr int NA = C::BM / RS, NB = C::BN / RS;
    static constexpr int LDA = KT * BKU;                  // A row stride (units)
    static constexpr int BAND = KT * 2048;                // one 32-row WTL band (units)
    static_assert(C::BM % RS == 0 && C::BN % RS == 0 && RS == 32, "chunk rows");
    const U* pa;      // A + (m0 + r0) LDA + 8 c
    const U* pz;      // a valid address for zero-filled chunks (A + 8 c)
    const U* pg;      // B rows of chunk 0 (q8gr: the gate band) + r0' 64 + 8 c
    const U* pu;      // q8gr: the up band
    uint32_t dst0;
    uint32_t amask;
    DEVI void init(const U* A, const U* B, int m0, int n0, int n1, int M) {
        const int tid = threadIdx.x;
        const int r0 = tid / C::CH, c = tid % C::CH;
        dst0 = (r0 * C::BK + ((c ^ swz_f<C::BK>(r0)) * 8)) * 2;
        pa = A + (size_t)(m0 + r0) * LDA + c * 8;
        pz = A + c * 8;
        amask = 0;
        if constexpr (!FULL) {
#pragma unroll
            for (int q = 0; q < NA; ++q) amask |= (m0 + r0 + q * RS < M ? 1u : 0u) << q;
        }
        if constexpr (C::EPI == 1 && C::WTN == 64) {          // chunk q: warp column q / 2, gate (q even) / up band
            pg = B + (size_t)(n0 >> 5) * BAND + r0 * 64 + c * 8;
            pu = B + (size_t)(n1 >> 5) * BAND + r0 * 64 + c * 8;
        } else if constexpr (C::EPI == 1 && C::WTN == 32) {   // chunk q: warp column q, rows 16 q + r0 % 16 of gate / up
            const int cc = r0 % 16;
            pg = B + (size_t)(((r0 >= 16) ? n1 : n0) >> 5) * BAND + cc * 64 + c * 8;
            pu = pg;
        } else {                                              // chunk q: band n0 / 32 + q, row r0
            static_assert(C::EPI == 2, "tile kind");
            pg = B + (size_t)(n0 >> 5) * BAND + r0 * 64 + c * 8;
            pu = pg;
        }
    }
    DEVI void issue(uint32_t sA, uint32_t sB, int kt) {
        issue_a(sA, kt);
        issue_b(sB, kt);
    }
    DEVI void issue_a(uint32_t sA, int kt) {
#pragma unroll
        for (int q = 0; q < NA; ++q) {
            const uint32_t d = sA + dst0 + q * RS * C::BK * 2;
            if constexpr (FULL) {
                cp_async16(d, pa + q * RS * LDA + kt * BKU, true);
            } else {
                const bool ok = (amask >> q) & 1;
                cp_async16(d, ok ? pa + q * RS * LDA + kt * BKU : pz, ok);
            }
        }
    }
    DEVI void issue_b(uint32_t sB, int kt) {
#pragma unroll
        for (int q = 0; q < NB; ++q) {
            const U* src;
            if constexpr (C::EPI == 1 && C::WTN == 64) src = ((q & 1) ? pu : pg) + (q / 2) * BAND;
            else if constexpr (C::EPI == 1 && C::WTN == 32) src = pg + (q / 2) * BAND + (q & 1) * 16 * 64;
            else src = pg + q * BAND;
            cp_async16(sB + dst0 + q * RS * C::BK * 2, src + kt * 2048, true);
        }
    }
};

#ifdef Q8H2_TM   // measurement only: phase marks of warp 0 lane 0 (0 tile start, 1 mainloop done, 2 staged, 3 tile done)
#define TM_MARK(P)                                                                                                 \
    if (threadIdx.x == 0) {                                                                                        \
        const long long c_ = clock64();                                                                            \
        if ((P) > 0)                                                                                               \
            atomicAdd((unsigned long long*)&tm_acc[blockIdx.x][(P)], (unsigned long long)(c_ - tm_last[blockIdx.x])); \
        else atomicAdd((unsigned long long*)&tm_acc[blockIdx.x][0], 1ull);                                         \
        tm_last[blockIdx.x] = c_;                                                                                  \
    }
#endif
struct TileId {
    int kind, m0, nt;   // kind 1 = a 128-row tile (CB), 0 = a tail tile (CS), 2 = a 64-row half of a 128-row tile (CP)
};
// rgh's hetero tile order: b < tb big tiles (m fastest), then the tail tiles of rows [mb, M); TS (ls > 0): the last ls
// big tiles move behind the tail tiles as 2 ls 64-row halves (the same rows and columns, finer work for the last round)
template <class CB, class CS>
DEVI TileId decode(const ArgsP& a, int b) {
    constexpr int bnc = CB::EPI == 1 ? CB::BN / 2 : CB::BN;   // output columns per tile
    const int nc = (CB::EPI == 1 ? a.N / 2 : a.N) / bnc;
    const int mtb = a.mb / CB::BM;
    const int tb = mtb * nc;
    const int nbig = tb - a.ls;
    if (b < nbig) return {1, (b % mtb) * CB::BM, b / mtb};
    b -= nbig;
    const int mts = (a.M - a.mb + CS::BM - 1) / CS::BM;
    const int nsm = mts * nc;
    if (b < nsm) return {0, a.mb + (b % mts) * CS::BM, b / mts};
    b -= nsm;
    const int k = nbig + (b >> 1);   // the big tile, rows (k % mtb) 128 + 64 (b & 1)
    return {2, (k % mtb) * CB::BM + (b & 1) * 64, k / mtb};
}

template <class C, int KT, bool FULL>
DEVI void loader_for(LoaderW<C, KT, FULL>& ld, const ArgsP& a, const TileId& t) {
    const int nbase = C::EPI == 1 ? t.nt * (C::BN / 2) : t.nt * C::BN;
    const int n1 = C::EPI == 1 ? a.N / 2 + nbase : -1;
    ld.init(reinterpret_cast<const U*>(a.A), reinterpret_cast<const U*>(a.B), t.m0, nbase, n1, a.M);
}

// k-tile 0 of tile b into stage 0 (its own loader: the tile kind may differ from the running one)
template <class CB, class CS, class CP, int KT>
DEVI void prefetch0(const ArgsP& a, int b, uint32_t s0) {
    const TileId t = decode<CB, CS>(a, b);
    auto go = [&](auto& ld) {
        loader_for(ld, a, t);
        ld.issue(st_a(s0, 0), st_b(s0, 0), 0);
    };
    if (t.kind == 1) {
        LoaderW<CB, KT, true> ld;
        go(ld);
    } else if (t.kind == 0) {
        LoaderW<CS, KT, false> ld;
        go(ld);
    } else {
        LoaderW<CP, KT, true> ld;   // a half of a 128-row tile: every row < mb <= M
        go(ld);
    }
}

// the tile's dequant inputs into smem (issued with k-tile 1, so they land under the mainloop): sa rows [m0, m0 + BM)
// (rows >= M zero-filled: rgh's dequant uses 0 there), sw / bias of the tile's 256 B rows in local column order
// (q8gr: gate columns o0 + [0, 128) then up columns H + o0 + [0, 128)); a null bias zero-filled (= dequant's 0)
template <class C>
DEVI void issue_scales(const ArgsP& a, const TileId& t, uint32_t sc) {
    const int tid = threadIdx.x;
    if (tid < C::BM / 4) {
        const int r = t.m0 + tid * 4;
        const int n = a.M - r;
        const int bytes = n >= 4 ? 16 : (n > 0 ? 4 * n : 0);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sc + tid * 16),
                     "l"(bytes ? a.sa + r : a.sa), "r"(bytes));
    } else if (tid >= 64 && tid < 192) {   // 64 chunks of sw, 64 of bias
        const int k = tid - 64, bi = k >= 64, lc = (k & 63) * 4;
        const int col = C::EPI == 1 ? (lc < 128 ? t.nt * 128 + lc : a.N / 2 + t.nt * 128 + lc - 128) : t.nt * 256 + lc;
        const float* src = bi ? a.bias : a.sw;
        const bool ok = src != nullptr;
        cp_async16(sc + 512 + bi * 1024 + lc * 4, ok ? src + col : a.sw, ok);
    }
}

// q8::dequant with sa / sw / bias read from the smem copy: fa = fmaf(acc, sa[r] sw[n], bias[n]) (the same lines)
template <class C>
DEVI void dequant_s(const int (&ia)[C::MT][C::NT][4], float (&fa)[C::MT][C::NT][4], const unsigned char* smem) {
    const float* ssa = reinterpret_cast<const float*>(smem + SCL);
    const float* ssw = ssa + 128;
    const float* sbi = ssa + 384;
    float rs[C::MT][2];
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) rs[i][h] = ssa[row_of<C>(i, h, 0)];
#pragma unroll
    for (int j = 0; j < C::NT; ++j) {
        const int c0 = col_of<C>(j, 0, 0, 128);   // the local column (q8gr: up columns at 128 +)
        const float2 w = *reinterpret_cast<const float2*>(ssw + c0);
        const float2 b = *reinterpret_cast<const float2*>(sbi + c0);
#pragma unroll
        for (int i = 0; i < C::MT; ++i)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float s = rs[i][q >> 1] * ((q & 1) ? w.y : w.x);
                fa[i][j][q] = fmaf((float)ia[i][j][q], s, (q & 1) ? b.y : b.x);
            }
    }
}

// q8h epilogue: the smem dequant; the bf16 tile (pack_bf2 = store_tile's rounding) staged in A1 | B1 (48 KB; B0 | A0
// hold the next tile's k-tile 0) as 512 B rows (16-byte chunk c of row r at chunk c ^ (r & 7): conflict-free 4-byte
// writes from the MMA layout and row reads), a tile of more than 96 rows in phases of its warp rows wm (64 rows each);
// each phase read back one 512 B row per warp and stored with 16-byte stores of rows < M (store_tile's coalescing)
template <class C>
DEVI void epi_h(const ArgsP& a, const int (&ia)[C::MT][C::NT][4], int m0, int n0, unsigned char* smem) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN;
    float fa[C::MT][C::NT][4];
    dequant_s<C>(ia, fa, smem);
    unsigned char* T = smem + EPI;
    constexpr int PH = C::BM * 512 > 49152 ? C::WM : 1;   // phases
    constexpr int PR = C::BM / PH;                        // rows per phase
    __nv_bfloat16* O = reinterpret_cast<__nv_bfloat16*>(a.out);
#pragma unroll
    for (int ph = 0; ph < PH; ++ph) {
        if (ph > 0) __syncthreads();   // the previous phase's rows read back
        if (PH == 1 || wm == ph) {
#pragma unroll
            for (int i = 0; i < C::MT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int rl = row_of<C>(i, h, 0) - ph * PR;
#pragma unroll
                    for (int j = 0; j < C::NT; ++j) {
                        const int cl = col_of<C>(j, 0, 0, 0);
                        *reinterpret_cast<uint32_t*>(T + rl * 512 + (((cl >> 3) ^ (rl & 7)) << 4) + (cl & 7) * 2) =
                            pack_bf2(fa[i][j][2 * h], fa[i][j][2 * h + 1]);
                    }
                }
        }
        __syncthreads();
        constexpr int NC = PR * 32 / C::THREADS;
#pragma unroll
        for (int k = 0; k < NC; ++k) {
            const int rl = k * (C::THREADS / 32) + warp;
            const uint4 v = *reinterpret_cast<const uint4*>(T + rl * 512 + ((lane ^ (rl & 7)) << 4));
            const int r = m0 + ph * PR + rl;
            if (r < a.M) *reinterpret_cast<uint4*>(O + (size_t)r * a.ldo + n0 + lane * 8) = v;
        }
    }
}

// q8gr epilogue: the smem dequant; the LUT SiLU line (q8k epi_g_rotK, LV & 1) -> bf16 staging T [BM][128] in stage 1
// (swizzled, tsw); the per-column centring mu and the rotation sign folded into is (round(v (-s)) = -round(v s):
// rot_sign(c, (x - mu) is) == (x - mu) (+-is) bit for bit) at T + 32 KB. Rotation phase: 4 lanes per row (lane q of a
// row holds its columns 32 q .. 32 q + 31), 8 rows per warp pass: rgh's 128-point butterflies in his stage order
// (ascending index bits; every stage low' = low + high, high' = low - high): bits 0 .. 4 in registers
// (q8::fwht_reg<32>), bits 5, 6 across the 4 lanes (one exact FFMA each: fma(v, -1, o) = o - v, fma(v, 1, o) =
// v + o); the row amax over the 4 lanes; then rgh's 127 / m, hs = m / 127 ROT_RS and rn(y 127 / m), 32 bytes per lane
template <class C, bool STAGE_ONLY>
DEVI void epi_gr(const ArgsP& a, const int (&ia)[C::MT][C::NT][4], int m0, int o0, unsigned char* smem) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int t4 = lane % 4;
    constexpr int NH = C::NT / 2;
    const int H = a.N / 2;
    unsigned char* T = smem + EPI;
    float* smu = reinterpret_cast<float*>(T + 32768);   // [128] centres, [128] signed scales
    const bool cen = a.mu != nullptr;
    if (threadIdx.x < 128) {
        const int c = threadIdx.x;
        smu[c] = cen ? a.mu[o0 + c] : 0.f;
        smu[128 + c] = rot_sign(c, cen ? a.is[o0 + c] : 1.f);
    }
    float fa[C::MT][C::NT][4];
    dequant_s<C>(ia, fa, smem);
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int rl = wm * C::WTM + i * 16 + lane / 4 + h * 8;
#pragma unroll
            for (int j = 0; j < NH; ++j) {
                // bf16 roundings two at a time (F2FP.PACK_AB, 4x the rate of the single F2F; each element RN as
                // __float2bfloat16_rn): g, u, then x = bf16(s u) packed as the staged word
                const __nv_bfloat162 g2 = __floats2bfloat162_rn(fa[i][j][2 * h], fa[i][j][2 * h + 1]);
                const __nv_bfloat162 u2 = __floats2bfloat162_rn(fa[i][j + NH][2 * h], fa[i][j + NH][2 * h + 1]);
                const float s0 = __bfloat162float(__ushort_as_bfloat16(__ldg(&a.lut[__bfloat16_as_ushort(g2.x)])));
                const float s1 = __bfloat162float(__ushort_as_bfloat16(__ldg(&a.lut[__bfloat16_as_ushort(g2.y)])));
                const __nv_bfloat162 x2 =
                    __floats2bfloat162_rn(s0 * __bfloat162float(u2.x), s1 * __bfloat162float(u2.y));
                const int cl = wn * (C::WTN / 2) + j * 8 + t4 * 2;
                *reinterpret_cast<__nv_bfloat162*>(T + rl * 256 + (tsw(cl >> 3, rl) << 4) + (cl & 7) * 2) = x2;
            }
        }
    __syncthreads();
#ifdef Q8H2_TM
    TM_MARK(2);
#endif
    if constexpr (STAGE_ONLY) return;
    const int g = o0 / 128;
    const int q = lane & 3;
    float mu[32], si[32];   // this lane's 32 centres / signed scales (registers: the ia / fa arrays are dead here)
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float4 m4 = *reinterpret_cast<const float4*>(smu + q * 32 + k * 4);
        const float4 s4 = *reinterpret_cast<const float4*>(smu + 128 + q * 32 + k * 4);
        mu[4 * k] = m4.x, mu[4 * k + 1] = m4.y, mu[4 * k + 2] = m4.z, mu[4 * k + 3] = m4.w;
        si[4 * k] = s4.x, si[4 * k + 1] = s4.y, si[4 * k + 2] = s4.z, si[4 * k + 3] = s4.w;
    }
    const float sg0 = (lane & 1) ? -1.f : 1.f, sg1 = (lane & 2) ? -1.f : 1.f;
    constexpr int NP = (C::BM + 63) / 64;   // warp passes of 8 rows
#pragma unroll 1
    for (int p = 0; p < NP; ++p) {
        const int rl = p * 64 + warp * 8 + lane / 4;
        if (p * 64 + warp * 8 >= C::BM) break;   // warp-uniform (BM % 8 == 0)
        float y[32];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint4 w = *reinterpret_cast<const uint4*>(T + rl * 256 + (tsw(q * 4 + k, rl) << 4));
            const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                y[k * 8 + 2 * t] = __uint_as_float(ww[t] << 16);
                y[k * 8 + 2 * t + 1] = __uint_as_float(ww[t] & 0xffff0000u);
            }
        }
#pragma unroll
        for (int e = 0; e < 32; ++e) y[e] = cen ? __fmul_rn(__fsub_rn(y[e], mu[e]), si[e]) : __fmul_rn(y[e], si[e]);
        fwht_reg<32>(y);
#pragma unroll
        for (int e = 0; e < 32; ++e) y[e] = __fmaf_rn(y[e], sg0, __shfl_xor_sync(0xffffffffu, y[e], 1));
#pragma unroll
        for (int e = 0; e < 32; ++e) y[e] = __fmaf_rn(y[e], sg1, __shfl_xor_sync(0xffffffffu, y[e], 2));
        float t[16];   // the row amax: an order-free max, as a tree
#pragma unroll
        for (int e = 0; e < 16; ++e) t[e] = fmaxf(fabsf(y[e]), fabsf(y[e + 16]));
#pragma unroll
        for (int w = 8; w > 0; w >>= 1)
#pragma unroll
            for (int e = 0; e < w; ++e) t[e] = fmaxf(t[e], t[e + w]);
        float m = t[0];
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        m = fmaxf(m, 1e-12f);
        const int r = m0 + rl;
        if (r < a.M) {
            const float inv = 127.f / m;
            if (q == 0) a.hs[(size_t)g * a.M + r] = m / 127.f * ROT_RS;
            uint32_t q4[8];
#pragma unroll
            for (int t8 = 0; t8 < 8; ++t8) {
                q4[t8] = 0;
#pragma unroll
                for (int e = 0; e < 4; ++e) q4[t8] |= (uint32_t)(__float2int_rn(y[4 * t8 + e] * inv) & 0xff) << (8 * e);
            }
            uint4* dst = reinterpret_cast<uint4*>(reinterpret_cast<int8_t*>(a.out) + (size_t)r * H + o0 + q * 32);
            dst[0] = make_uint4(q4[0], q4[1], q4[2], q4[3]);
            dst[1] = make_uint4(q4[4], q4[5], q4[6], q4[7]);
        }
    }
}

// one tile of kind C (FULL: every row < M): k-tile 0 already in flight into stage 0 (pre) or issued here; the int8
// mainloop (q8::mainloop's ring: waits, one barrier per k-tile, refill point, fragment double-buffering, MMA order);
// thread 0 takes the next tile from the counter at k-tile KT - 3 (so tiles go out in about the order CTAs finish
// them) and publishes it at KT - 2; the last k-tile issues the next tile's k-tile 0 into stage 0; then the epilogue
// (EK 0 = q8h, 1 = q8gr). PERSIST 2 = this; 1 (measurement form) = persistent without the k-tile 0 prefetch;
// 0 (measurement form) = one tile per CTA, nothing fetched or prefetched
template <class C, bool FULL, class CB, class CS, class CP, int EK, int KT, int PERSIST>
DEVI void do_tile(const ArgsP& a, const TileId& t, bool pre, volatile int* nslot, unsigned char* smem) {
    constexpr int KS = C::KS, MT = C::MT, NT = C::NT;
    static_assert(KT % 2 == 0 && KT >= 4, "an even k-tile count: every tile's k-tile 0 is in stage 0");
    const uint32_t s0 = smem_u32(smem);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    LoaderW<C, KT, FULL> ld;
    loader_for(ld, a, t);
    if (!pre) {
        ld.issue(st_a(s0, 0), st_b(s0, 0), 0);
        cp_commit();
    }
    int ia[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) ia[i][j][q] = 0;
    Frag<C> fr;
    fr.init(warp / C::WN, warp % C::WN, lane);
    cp_wait<0>();
    __syncthreads();
    fr.load(0, st_a(s0, 0), st_b(s0, 0), 0);
    auto mmas = [&](int cb) {
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int ii = 0; ii < MT; ++ii) {
                const int i = (j & 1) ? MT - 1 - ii : ii;
                mma_s8(ia[i][j], fr.af[cb][i], fr.bf[cb][j][0], fr.bf[cb][j][1]);
            }
    };
    // one k-tile kt in stage S (compile-time): TAG 1 = k-tile 0 (+ the tile's scales), 2 = KT - 3 (thread 0 takes the
    // next tile from the counter; its result is first used one k-tile later), 3 = KT - 2 (thread 0 publishes it); 0 =
    // plain. The special k-tiles are peeled, so the plain ones carry no per-k-tile conditions
    int nx = 0;
    auto ktile = [&](int kt, auto st, auto tag) {
        constexpr int S = decltype(st)::value, TAG = decltype(tag)::value;
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            if (ks == KS - 1) {
                cp_wait<0>();
                __syncthreads();
            }
            if (ks == 0) {
                ld.issue(st_a(s0, S ^ 1), st_b(s0, S ^ 1), kt + 1);
                if constexpr (TAG == 1) issue_scales<C>(a, t, s0 + SCL);
                if constexpr (TAG == 2 && PERSIST != 0)
                    if (threadIdx.x == 0) nx = (int)gridDim.x + atomicAdd(a.ctr, 1);
                if constexpr (TAG == 3 && PERSIST != 0)
                    if (threadIdx.x == 0) *nslot = nx;
            }
            const int ns = ks < KS - 1 ? S : S ^ 1;
            fr.load((ks + 1) & 1, st_a(s0, ns), st_b(s0, ns), ks < KS - 1 ? ks + 1 : 0);
            mmas(ks & 1);
            if (ks == 0) cp_commit();
        }
    };
    using I0 = std::integral_constant<int, 0>;
    using I1 = std::integral_constant<int, 1>;
    using I2 = std::integral_constant<int, 2>;
    using I3 = std::integral_constant<int, 3>;
    static_assert(KT >= 6 && (KT - 4) % 2 == 0, "k-tile 0, pairs of plain k-tiles, KT - 3, KT - 2, the last");
    ktile(0, I0{}, I1{});
#pragma unroll 1
    for (int kt = 1; kt < KT - 3; kt += 2) {
        ktile(kt, I1{}, I0{});
        ktile(kt + 1, I0{}, I0{});
    }
    ktile(KT - 3, I1{}, I2{});
    ktile(KT - 2, I0{}, I3{});
    // the last k-tile (stage 1): the next tile's k-tile 0 goes into stage 0 (free since the last barrier)
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
        if (ks == 0 && PERSIST == 2) {
            const int n = *nslot;
            if (n < a.nt) prefetch0<CB, CS, CP, KT>(a, n, s0);
            cp_commit();
        }
        if (ks < KS - 1) fr.load((ks + 1) & 1, st_a(s0, 1), st_b(s0, 1), ks + 1);
        mmas(ks & 1);
    }
    __syncthreads();   // every warp is past its last fragment load: stage 1 (A1 | B1) is the epilogue's
#ifdef Q8H2_TM
    TM_MARK(1);
#endif
    if constexpr (EK == 0) {
        epi_h<C>(a, ia, t.m0, t.nt * C::BN, smem);
    } else if constexpr (EK == 1) {
        epi_gr<C, false>(a, ia, t.m0, t.nt * (C::BN / 2), smem);
    } else if constexpr (EK == 2) {   // measurement only: the mainloop alone (its sums folded into one word)
        int x = 0;
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int q = 0; q < 4; ++q) x ^= ia[i][j][q];
        reinterpret_cast<int*>(a.hs)[blockIdx.x * C::THREADS + threadIdx.x] = x;
    } else {                          // measurement only: q8gr's dequant + SwiGLU + staging, no rotation phase
        epi_gr<C, true>(a, ia, t.m0, t.nt * (C::BN / 2), smem);
    }
}

template <class CB, class CS, class CP, int EK, int KT, int PERSIST>
DEVI void run(const ArgsP& a, unsigned char* smem) {
    volatile int* slot = reinterpret_cast<volatile int*>(smem + SLOT);
    int b = blockIdx.x, par = 0;
    bool pre = false;
    while (b < a.nt) {
        const TileId t = decode<CB, CS>(a, b);
#ifdef Q8H2_TM
        TM_MARK(0);
#endif
        if (t.kind == 1) do_tile<CB, true, CB, CS, CP, EK, KT, PERSIST>(a, t, pre, slot + par, smem);
        else if (t.kind == 0) do_tile<CS, false, CB, CS, CP, EK, KT, PERSIST>(a, t, pre, slot + par, smem);
        else do_tile<CP, true, CB, CS, CP, EK, KT, PERSIST>(a, t, pre, slot + par, smem);
#ifdef Q8H2_TM
        TM_MARK(3);
#endif
        if (PERSIST == 0) return;
        b = slot[par];
        par ^= 1;
        pre = PERSIST == 2;
    }
    if (threadIdx.x == 0 && atomicAdd(a.ctr + 1, 1) == (int)gridDim.x - 1) {   // the last CTA: counters back to 0
        a.ctr[0] = 0;
        a.ctr[1] = 0;
    }
}

}  // namespace q8h2

#define Q8H2_KERNEL(NAME, CB, CS, CP, EK, PERSIST)                                                                 \
    extern "C" __global__ void __launch_bounds__(CB::THREADS, 1) NAME(q8h2::ArgsP a) {                              \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        q8h2::run<CB, CS, CP, EK, 20, PERSIST>(a, smem_raw);                                                        \
    }

using P_S96x256 = q8::Cfg<96, 256, 64, 2, 4, 2, 2>;   // = q8.cu's Q8K block (ruj's s96 tail tile)
using W_S96x256 = q8::Cfg<96, 256, 64, 2, 4, 2, 1>;

// K = 2560 (20 k-tiles): every "in" family at H = 0
Q8H2_KERNEL(q8h2_h_s64x256, P_B128x256, P_S64x256, P_S64x256, 0, 2)
Q8H2_KERNEL(q8h2_h_s32x256, P_B128x256, P_S32x256, P_S64x256, 0, 2)
Q8H2_KERNEL(q8h2_h_s96x256, P_B128x256, P_S96x256, P_S64x256, 0, 2)
Q8H2_KERNEL(q8h2_gr_s64x256, W_B128x256, W_S64x256, W_S64x256, 1, 2)
Q8H2_KERNEL(q8h2_gr_s32x256, W_B128x256, W_S32x256, W_S64x256, 1, 2)
Q8H2_KERNEL(q8h2_gr_s96x256, W_B128x256, W_S96x256, W_S64x256, 1, 2)
#ifdef Q8H2_PROBE   // measurement forms: q8h2probe.cubin only (np = one tile per CTA, grid = tiles; p1 = no prefetch)
Q8H2_KERNEL(q8h2np_h_s64x256, P_B128x256, P_S64x256, P_S64x256, 0, 0)
Q8H2_KERNEL(q8h2np_gr_s64x256, W_B128x256, W_S64x256, W_S64x256, 1, 0)
Q8H2_KERNEL(q8h2p1_h_s64x256, P_B128x256, P_S64x256, P_S64x256, 0, 1)
Q8H2_KERNEL(q8h2p1_gr_s64x256, W_B128x256, W_S64x256, W_S64x256, 1, 1)
Q8H2_KERNEL(q8h2ml_gr_s64x256, W_B128x256, W_S64x256, W_S64x256, 2, 2)   // mainloop only
Q8H2_KERNEL(q8h2st_gr_s64x256, W_B128x256, W_S64x256, W_S64x256, 3, 2)   // + dequant / SwiGLU / staging
Q8H2_KERNEL(q8h2ml_h_s64x256, P_B128x256, P_S64x256, P_S64x256, 2, 2)
#endif
