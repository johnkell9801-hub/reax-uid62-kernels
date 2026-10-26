// q8ri.cu (rgh 10-25): RIN - the "in" families' int8 inputs rotated before the per-row scale, so the int8 range is not
// set by a few outlier channels and the protected bf16 channels can go (KH = 0: the q8h / q8g kernels' bf16 pass
// disappears). Each 128-channel group of the lo order (8 consecutive channels per thread, 16 threads per group) is
// rotated with ROT's R = diag(d) H_128 / sqrt(128) (q8.cu: ROT_NEG, rot_sign, fwht_reg, fwht_lanes; the 1 / sqrt(128)
// folded into the stored row scale); the family's weight rows carry the same R (q8.py InFam, rot), so the q8h / q8g /
// q8gr GEMMs and their epilogues are unchanged: (x R)(w R)^T = x w^T.
//   q8n_addnormr: q8n_addnorm (add + RMSNorm -> hn, x, int8 rows) with the rotation in the int8 prepare; x may be null
//                 (Q8-NX: the bf16 x of a family no exact op reads is not stored).
//   q8x_prepr:    the same prepare from a bf16 row x [R, 2560] (q8x_prep's arguments; one CTA of 320 threads per row).
// Standalone cubin: no kernel of q8.cubin / q8k.cubin / q8pb.cubin changes.
// Build: nvcc -O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -o q8ri.cubin q8ri.cu
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

namespace q8ri {
constexpr int D = 2560;
constexpr float ROT_RS = 0.08838834764831845f;   // 1 / sqrt(128)
// q8.cu's ROT signs: d_i = -1 where bit i is set (the fine families' R; RIN uses the same R)
__device__ __constant__ uint32_t ROT_NEG[4] = {0x09bab9abu, 0xfc955699u, 0xfc9121c0u, 0xafc1071cu};

// v -> H_128 (d o v) over the group of this thread's half-warp: v holds group positions 8 (t % 16) .. + 7; every lane
// of the warp must call it (inactive lanes with zeros)
DEVI void rot128(float (&v)[8], int t) {
    const int i0 = (t & 15) * 8;
#pragma unroll
    for (int e = 0; e < 8; ++e)
        if ((ROT_NEG[(i0 + e) >> 5] >> ((i0 + e) & 31)) & 1u) v[e] = -v[e];
#pragma unroll
    for (int st = 1; st < 8; st <<= 1)
#pragma unroll
        for (int e = 0; e < 8; ++e)
            if ((e & st) == 0) {
                const float a = v[e], b = v[e + st];
                v[e] = a + b;
                v[e + st] = a - b;
            }
    const int lane = t & 31;
#pragma unroll
    for (int m = 1; m <= 8; m <<= 1)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float o = __shfl_xor_sync(0xffffffffu, v[e], m);
            v[e] = (lane & m) ? o - v[e] : v[e] + o;
        }
}

// the rotated int8 prepare of the bf16 row xs (shared, 2560) by 320 threads: xq [r, Kl], sa[r] (amax / 127 / sqrt(128)),
// xh [r, KH] (the protected channels, unrotated; KH = 0 in RIN's use)
DEVI void prep_row(const __nv_bfloat16* xs, int r, int t, float* red, int8_t* __restrict__ xq, float* __restrict__ sa,
                   __nv_bfloat16* __restrict__ xh, const int* __restrict__ lo, const float* __restrict__ mu,
                   const float* __restrict__ is, int Kl, const int* __restrict__ hi, int KH) {
    const int c = t * 8;
    const bool act = c < Kl;
    float q[8];
    if (act) {
        const int4 i0 = *reinterpret_cast<const int4*>(lo + c), i1 = *reinterpret_cast<const int4*>(lo + c + 4);
        const int ii[8] = {i0.x, i0.y, i0.z, i0.w, i1.x, i1.y, i1.z, i1.w};
        const float4 m0 = *reinterpret_cast<const float4*>(mu + c), m1 = *reinterpret_cast<const float4*>(mu + c + 4);
        const float4 s0 = *reinterpret_cast<const float4*>(is + c), s1 = *reinterpret_cast<const float4*>(is + c + 4);
        const float mm[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
        const float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) q[e] = (__bfloat162float(xs[ii[e]]) - mm[e]) * sv[e];
    } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) q[e] = 0.f;
    }
    rot128(q, t);   // Kl % 128 == 0: a group is all-active or all-inactive
    float amax = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf(q[e]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    __syncthreads();
    if ((t & 31) == 0) red[t >> 5] = amax;
    __syncthreads();
    amax = red[0];
#pragma unroll
    for (int w = 1; w < 10; ++w) amax = fmaxf(amax, red[w]);
    amax = fmaxf(amax, 1e-12f);
    const float inv = 127.f / amax;
    if (t == 0) sa[r] = amax / 127.f * ROT_RS;
    if (act) {
        uint32_t lo4 = 0, hi4 = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) lo4 |= (uint32_t)(__float2int_rn(q[e] * inv) & 0xff) << (8 * e);
#pragma unroll
        for (int e = 0; e < 4; ++e) hi4 |= (uint32_t)(__float2int_rn(q[4 + e] * inv) & 0xff) << (8 * e);
        *reinterpret_cast<uint2*>(xq + (size_t)r * Kl + c) = make_uint2(lo4, hi4);
    }
    for (int j = t; j < KH; j += 320) xh[(size_t)r * KH + j] = xs[hi[j]];
}
}  // namespace q8ri

// q8n_addnorm's add + RMSNorm (same arithmetic: hn = bf16(h + y), x = bf16(bf16-row * rsqrt(mean sq + eps) * w)) with
// q8ri's rotated prepare; part / P / y as q8n_addnorm (P > 0: y = bf16(sum of the P fp32 partials)); x null: not stored
extern "C" __global__ void __launch_bounds__(320) q8n_addnormr(
    const float* __restrict__ part, int P, const __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ h,
    const float* __restrict__ w1p, float eps, __nv_bfloat16* __restrict__ hn, __nv_bfloat16* __restrict__ x, int M,
    int8_t* __restrict__ xq, float* __restrict__ sa, __nv_bfloat16* __restrict__ xh, const int* __restrict__ lo,
    const float* __restrict__ mu, const float* __restrict__ is, int Kl, const int* __restrict__ hi, int KH) {
    constexpr int D = q8ri::D;
    const int r = blockIdx.x, t = threadIdx.x, c = t * 8;
    __shared__ float red[10];
    __shared__ __align__(16) __nv_bfloat16 xs[D];
    float v[8];
    {
        const uint4 hr = *reinterpret_cast<const uint4*>(h + (size_t)r * D + c);
        const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(&hr);
        float yy[8];
        if (P > 0) {
            const float* pr = part + (size_t)r * D + c;
            float4 a0 = *reinterpret_cast<const float4*>(pr), a1 = *reinterpret_cast<const float4*>(pr + 4);
            for (int p = 1; p < P; ++p) {
                const float* q = pr + (size_t)p * M * D;
                const float4 b0 = *reinterpret_cast<const float4*>(q), b1 = *reinterpret_cast<const float4*>(q + 4);
                a0.x += b0.x; a0.y += b0.y; a0.z += b0.z; a0.w += b0.w;
                a1.x += b1.x; a1.y += b1.y; a1.z += b1.z; a1.w += b1.w;
            }
            const float s[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
            for (int e = 0; e < 8; ++e) yy[e] = __bfloat162float(__float2bfloat16_rn(s[e]));
        } else {
            const uint4 yr = *reinterpret_cast<const uint4*>(y + (size_t)r * D + c);
            const __nv_bfloat16* yb = reinterpret_cast<const __nv_bfloat16*>(&yr);
#pragma unroll
            for (int e = 0; e < 8; ++e) yy[e] = __bfloat162float(yb[e]);
        }
        uint4 ho;
        __nv_bfloat16* hob = reinterpret_cast<__nv_bfloat16*>(&ho);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            hob[e] = __float2bfloat16_rn(__bfloat162float(hb[e]) + yy[e]);
            v[e] = __bfloat162float(hob[e]);
        }
        *reinterpret_cast<uint4*>(hn + (size_t)r * D + c) = ho;
    }
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) ss = fmaf(v[e], v[e], ss);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((t & 31) == 0) red[t >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < 10; ++w) tot += red[w];
    const float rr = rsqrtf(tot * (1.0f / D) + eps);
    {
        const float4 w0 = *reinterpret_cast<const float4*>(w1p + c), w1 = *reinterpret_cast<const float4*>(w1p + c + 4);
        const float ww[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        uint4 xo;
        __nv_bfloat16* xob = reinterpret_cast<__nv_bfloat16*>(&xo);
#pragma unroll
        for (int e = 0; e < 8; ++e) xob[e] = __float2bfloat16_rn((v[e] * rr) * ww[e]);
        if (x != nullptr) *reinterpret_cast<uint4*>(x + (size_t)r * D + c) = xo;
        *reinterpret_cast<uint4*>(xs + c) = xo;
    }
    __syncthreads();   // xs complete (the gather reads any channel); red reused below after a second barrier
    q8ri::prep_row(xs, r, t, red, xq, sa, xh, lo, mu, is, Kl, hi, KH);
}

// the rotated prepare of a bf16 row: x [R, ldx] -> xq [R, Kl], sa [R], xh [R, KH]; grid R, 320 threads (q8x_prep order
// of arguments without its ldq / M / G)
extern "C" __global__ void __launch_bounds__(320) q8x_prepr(const __nv_bfloat16* __restrict__ x, int ldx,
                                                            const int* __restrict__ lo, const float* __restrict__ mu,
                                                            const float* __restrict__ is, int8_t* __restrict__ xq,
                                                            float* __restrict__ sa, int Kl,
                                                            const int* __restrict__ hi, int KH,
                                                            __nv_bfloat16* __restrict__ xh) {
    const int r = blockIdx.x, t = threadIdx.x;
    __shared__ float red[10];
    __shared__ __align__(16) __nv_bfloat16 xs[q8ri::D];
    *reinterpret_cast<uint4*>(xs + t * 8) = *reinterpret_cast<const uint4*>(x + (size_t)r * ldx + t * 8);
    __syncthreads();
    q8ri::prep_row(xs, r, t, red, xq, sa, xh, lo, mu, is, Kl, hi, KH);
}
