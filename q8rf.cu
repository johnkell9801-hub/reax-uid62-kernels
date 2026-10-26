// q8rf.cu (rgh 10-25): ROTF - RIN with the FULL-row rotation of the in families' int8 inputs: after RIN's per-group
// R_128 = diag(d) H_128 / sqrt(128) (q8ri.cu rot128), the 20 groups of a 2560-channel row are mixed by the Paley
// Hadamard H_20 (q = 19): out[g][p] = sum_j H20[g][j] y[j][p] (j = 0 .. 19 in order, fp32), the 1 / sqrt(20) folded
// into the stored row scale with RIN's 1 / sqrt(128) (sa = amax / 127 / sqrt(2560)). The family's weight rows carry
// the same R_full = blockdiag(R_128) (H20^T (x) I_128) / sqrt(20) (q8.py / q8derive.py, ROTF), so the GEMMs and
// their epilogues are unchanged: (x R)(w R)^T = x w^T. An outlier channel's mass now spreads over all 2560 int8
// columns instead of its 128-column group.
//   q8n_addnormf: q8ri's q8n_addnormr (add + RMSNorm, same arithmetic) with the full-row rotated prepare
//   q8x_prepf:    q8ri's q8x_prepr with the full-row rotated prepare
// Build: nvcc -O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -o q8rf.cubin q8rf.cu
#include "q8ri.cu"

namespace q8rf {
constexpr float ROT_RS = 0.019764235376052372f;   // 1 / sqrt(2560)

// q8ri::prep_row with the H20 group mix after rot128 (Kl = 2560: every thread active); fb: shared float [2560]
DEVI void prep_row(const __nv_bfloat16* xs, int r, int t, float* red, float* fb, int8_t* __restrict__ xq,
                   float* __restrict__ sa, const int* __restrict__ lo, const float* __restrict__ mu,
                   const float* __restrict__ is) {
    const int c = t * 8;
    float q[8];
    {
        const int4 i0 = *reinterpret_cast<const int4*>(lo + c), i1 = *reinterpret_cast<const int4*>(lo + c + 4);
        const int ii[8] = {i0.x, i0.y, i0.z, i0.w, i1.x, i1.y, i1.z, i1.w};
        const float4 m0 = *reinterpret_cast<const float4*>(mu + c), m1 = *reinterpret_cast<const float4*>(mu + c + 4);
        const float4 s0 = *reinterpret_cast<const float4*>(is + c), s1 = *reinterpret_cast<const float4*>(is + c + 4);
        const float mm[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
        const float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) q[e] = (__bfloat162float(xs[ii[e]]) - mm[e]) * sv[e];
    }
    q8ri::rot128(q, t);
    *reinterpret_cast<float4*>(fb + c) = make_float4(q[0], q[1], q[2], q[3]);
    *reinterpret_cast<float4*>(fb + c + 4) = make_float4(q[4], q[5], q[6], q[7]);
    __syncthreads();
    if (t < 128) {   // thread t mixes group position t of all 20 groups in registers (each value read / written once)
        // H20 = Paley I (q = 19): row g is -1 at column j where bit j is set (compile-time: pure FADD / FSUB)
        constexpr uint32_t H20M[20] = {0x0u,     0x9ea19u, 0x3d433u, 0x7a865u, 0xf50c9u, 0xea193u, 0xd4327u,
                                       0xa864fu, 0x50c9fu, 0xa193du, 0x4327bu, 0x864f5u, 0xc9ebu,  0x193d5u,
                                       0x327a9u, 0x64f51u, 0xc9ea1u, 0x93d43u, 0x27a87u, 0x4f50du};
        float v[20], o[20];
#pragma unroll
        for (int j = 0; j < 20; ++j) v[j] = fb[j * 128 + t];
#pragma unroll
        for (int g = 0; g < 20; ++g) {
            const uint32_t m = H20M[g];
            o[g] = 0.f;
#pragma unroll
            for (int j = 0; j < 20; ++j) o[g] = ((m >> j) & 1u) ? o[g] - v[j] : o[g] + v[j];
        }
#pragma unroll
        for (int g = 0; g < 20; ++g) fb[g * 128 + t] = o[g];
    }
    __syncthreads();
    {
        const float4 a = *reinterpret_cast<const float4*>(fb + c), b = *reinterpret_cast<const float4*>(fb + c + 4);
        q[0] = a.x; q[1] = a.y; q[2] = a.z; q[3] = a.w; q[4] = b.x; q[5] = b.y; q[6] = b.z; q[7] = b.w;
    }
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
    uint32_t lo4 = 0, hi4 = 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) lo4 |= (uint32_t)(__float2int_rn(q[e] * inv) & 0xff) << (8 * e);
#pragma unroll
    for (int e = 0; e < 4; ++e) hi4 |= (uint32_t)(__float2int_rn(q[4 + e] * inv) & 0xff) << (8 * e);
    *reinterpret_cast<uint2*>(xq + (size_t)r * 2560 + c) = make_uint2(lo4, hi4);
}
}  // namespace q8rf

// q8ri's q8n_addnormr (the same add + RMSNorm op for op: hn, x identical) with q8rf's full-row rotated prepare;
// the argument list is q8n_addnormr's (Kl must be 2560, KH 0)
extern "C" __global__ void __launch_bounds__(320) q8n_addnormf(
    const float* __restrict__ part, int P, const __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ h,
    const float* __restrict__ w1p, float eps, __nv_bfloat16* __restrict__ hn, __nv_bfloat16* __restrict__ x, int M,
    int8_t* __restrict__ xq, float* __restrict__ sa, __nv_bfloat16* __restrict__ xh, const int* __restrict__ lo,
    const float* __restrict__ mu, const float* __restrict__ is, int Kl, const int* __restrict__ hi, int KH) {
    constexpr int D = q8ri::D;
    const int r = blockIdx.x, t = threadIdx.x, c = t * 8;
    __shared__ float red[10];
    __shared__ __align__(16) __nv_bfloat16 xs[D];
    __shared__ __align__(16) float fb[D];
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
    q8rf::prep_row(xs, r, t, red, fb, xq, sa, lo, mu, is);
}

// q8ri's q8x_prepr with q8rf's full-row rotated prepare (q8x_prepr's argument list; Kl 2560, KH 0)
extern "C" __global__ void __launch_bounds__(320) q8x_prepf(const __nv_bfloat16* __restrict__ x, int ldx,
                                                            const int* __restrict__ lo, const float* __restrict__ mu,
                                                            const float* __restrict__ is, int8_t* __restrict__ xq,
                                                            float* __restrict__ sa, int Kl,
                                                            const int* __restrict__ hi, int KH,
                                                            __nv_bfloat16* __restrict__ xh) {
    const int r = blockIdx.x, t = threadIdx.x;
    __shared__ float red[10];
    __shared__ __align__(16) __nv_bfloat16 xs[q8ri::D];
    __shared__ __align__(16) float fb[q8ri::D];
    *reinterpret_cast<uint4*>(xs + t * 8) = *reinterpret_cast<const uint4*>(x + (size_t)r * ldx + t * 8);
    __syncthreads();
    q8rf::prep_row(xs, r, t, red, fb, xq, sa, lo, mu, is);
}
