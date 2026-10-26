// Q8F2 (rgh 10-25, ours; S1ENG_Q8F2): the fine-group int8 GEMMs (down_proj K 9216 = 72 groups, GDN out_proj /
// attention o_proj K 4096 = 32 groups, N 2560) with 64- and 96-row big tiles, picked per row count R. e41 runs every
// R through pjs's Q8-PB q8f_b128x128_<small>_pbd (q8pb.cu: rgh's mainloop_fine + bf16x2 register stores, one split),
// whose 128 x 128 CTA walks all of K in ~75 us (down) / ~36 us (out) whatever R is: at the served R (time-weighted
// p10 / p50 / p90 448 / 896 / 1280) the grid is 20 x ceil(R / 128) = 20 .. 220 CTAs on 188 SMs, so below ~900 rows
// most SMs idle for one full CTA time and above 1152 rows a second round of 75 us follows. A 64 x 128 CTA takes
// ~46 us (down) / ~23 us (out) and a 96 x 128 CTA ~60 / ~29 us (less reuse per streamed byte, but at these grids the
// k-loop is latency-bound, not bandwidth-bound), so with 20 x ceil(R / 64) or 20 x ceil(R / 96) CTAs the launch ends
// sooner whenever that grid still fits the SMs (or turns 2 rounds of 75 us into 2 rounds of 60 us).
// EXACT: every kernel here is pjs's tile_fpb (DIRECT) with rgh's mainloop_fine (q8.cu, included unchanged) on another
// tile configuration. One thread computes each output element with the same chain
//   fa = 0; for g = 0 .. G-1: fa = fmaf((float)acc_g, sa[g][r] * sw[g][n], fa); fa += bias[n]; out = bf16(fa)
// (acc_g = the exact int32 sum of the group's 128 int8 products, any order); the tile height only decides which CTA /
// thread holds the element, so the bf16 rows are byte-identical to _pbd's at every R (gated at start-up, q8f2.py).
// Kernels (ArgsF; Ks = K, wfull = 0 or 1; rows [0, mb) big tiles, m fastest, then ONE row of small tiles, M - mb <=
// the small tile's height):
//   q8f2_b64x128_s{32,64}x128      64-row big tiles (rgh's s64x128 configuration: 2 x 4 warps of 32 x 32, 3 stages)
//   q8f2_b96x128_s{32,64,96}x128   96-row big tiles (pjs's Q8-S96 configuration: 2 x 4 warps of 48 x 32, 3 stages)
#include "q8.cu"

namespace q8 {
namespace f2 {

// pjs's store_rows_bf16 (q8pb.cu): bf16x2 stores of rows < M straight from the accumulators
template <class C>
DEVI void store_rows(const float (&v)[C::MT][C::NT][4], __nv_bfloat16* __restrict__ P, int M, int N, int m0, int n0) {
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = row_of<C>(i, h, m0);
            if (r >= M) continue;
            __nv_bfloat16* row = P + (size_t)r * N;
#pragma unroll
            for (int j = 0; j < C::NT; ++j)
                *reinterpret_cast<uint32_t*>(row + col_of<C>(j, 0, n0, 0)) = pack_bf2(v[i][j][2 * h], v[i][j][2 * h + 1]);
        }
}

// pjs's tile_fpb (DIRECT): rgh's mainloop_fine, rgh's bias add, the bf16x2 register store
template <class C>
DEVI void tile(const ArgsF& a, int m0, int nt, U* smem) {
    const int n0 = nt * C::BN;
    float fa[C::MT][C::NT][4];
    mainloop_fine<C>(reinterpret_cast<const U*>(a.A), reinterpret_cast<const U*>(a.B), a.M, a.N, a.Ks / 2, a.lda / 2,
                     a.ldb / 2, m0, n0, smem, fa, a.sa, a.sw, a.wfull);
    if (a.bias != nullptr) {
#pragma unroll
        for (int j = 0; j < C::NT; ++j) {
            const float2 b = *reinterpret_cast<const float2*>(a.bias + col_of<C>(j, 0, n0, 0));
#pragma unroll
            for (int i = 0; i < C::MT; ++i)
#pragma unroll
                for (int q = 0; q < 4; ++q) fa[i][j][q] += (q & 1) ? b.y : b.x;
        }
    }
    store_rows<C>(fa, reinterpret_cast<__nv_bfloat16*>(a.out), a.M, a.N, m0, n0);
}

// rgh's hetero_f tile map: rows [0, mb) big tiles (CTAs 0 .. Tb-1, m fastest), rows [mb, M) small tiles
template <class CB, class CS>
DEVI void hetero(const ArgsF& a, U* smem) {
    const int mtb = a.mb / CB::BM;
    const int tb = mtb * (a.N / CB::BN);
    int b = blockIdx.x;
    if (b < tb) {
        tile<CB>(a, (b % mtb) * CB::BM, b / mtb, smem);
    } else {
        b -= tb;
        const int mts = (a.M - a.mb + CS::BM - 1) / CS::BM;
        tile<CS>(a, a.mb + (b % mts) * CS::BM, b / mts, smem);
    }
}

}  // namespace f2
}  // namespace q8

#define Q8F2_KERNEL(NAME, CB, CS)                                                                                  \
    extern "C" __global__ void __launch_bounds__(CB::THREADS, 1) NAME(q8::ArgsF a) {                               \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        q8::f2::hetero<CB, CS>(a, reinterpret_cast<q8::U*>(smem_raw));                                             \
    }
// (BM, BN, BK units, warps M, warps N, stages, epilogue): rgh's s64x128 / s32x128 and pjs's s96x128 configurations
using F2_B64x128 = q8::Cfg<64, 128, 64, 2, 4, 3, 2>;
using F2_B96x128 = q8::Cfg<96, 128, 64, 2, 4, 3, 2>;
using F2_S32x128 = q8::Cfg<32, 128, 64, 1, 8, 3, 2>;
Q8F2_KERNEL(q8f2_b64x128_s32x128, F2_B64x128, F2_S32x128)
Q8F2_KERNEL(q8f2_b64x128_s64x128, F2_B64x128, F2_B64x128)
Q8F2_KERNEL(q8f2_b96x128_s32x128, F2_B96x128, F2_S32x128)
Q8F2_KERNEL(q8f2_b96x128_s64x128, F2_B96x128, F2_B64x128)
Q8F2_KERNEL(q8f2_b96x128_s96x128, F2_B96x128, F2_B96x128)
