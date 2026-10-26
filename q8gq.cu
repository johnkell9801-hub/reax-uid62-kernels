// q8gq.cu (rgh, 25 Oct 2026; Q8 DERIVE, q8derive.py): the column loop of one 128-column GPTQ block in one launch.
// GPTQ's inner loop (q8derive.gptq_ref: for each column i of the block: q = clamp(rn(w_i / b), -127, 127); err =
// (w_i - q b) / Hi[i][i]; W1[:, i:] -= err Hi[i, i:]) is independent per row: one warp per row keeps the row's 128
// columns in registers (lane l: columns l, l + 32, l + 64, l + 96), the owner lane of column i broadcasts w_i, every lane
// computes q / err itself and updates its own columns. Every operation is the IEEE single-rounding op the torch loop runs
// (__fmul_rn / __fsub_rn / __fdiv_rn: never contracted into an FMA; rintf = torch.round's half-to-even; fminf / fmaxf =
// torch.clamp), so Q / E equal the torch loop's bit for bit (checked at start-up on random rows, q8derive.gq_check).
// Build: nvcc -O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -o q8gq.cubin q8gq.cu
#include <cstdint>

extern "C" __global__ void __launch_bounds__(256) q8gq_block(
    const float* __restrict__ W, int ldw,     // W[:, i1:i1 + 128] (row stride ldw): the block's current columns
    const float* __restrict__ Hi, int ldh,    // Hinv[i1:i1 + 128, i1:i1 + 128] (row stride ldh), upper Cholesky factor
    const float* __restrict__ b,              // [N] the row scales of this block
    float* __restrict__ Q, int ldq,           // Q[:, i1:i1 + 128] (row stride ldq): the integer values (as float)
    float* __restrict__ E,                    // [N, 128] contiguous: the scaled errors (E1)
    int N)
{
    const int row = (int)((blockIdx.x * (unsigned)blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= N) return;
    const float* w = W + (size_t)row * ldw;
    float r[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) r[k] = w[lane + 32 * k];
    const float br = b[row];
    float* qrow = Q + (size_t)row * ldq;
    float* erow = E + (size_t)row * 128;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
#pragma unroll
        for (int o = 0; o < 32; ++o) {
            const int i = s * 32 + o;
            const float wi = __shfl_sync(0xffffffffu, r[s], o);
            const float q = fminf(fmaxf(rintf(__fdiv_rn(wi, br)), -127.0f), 127.0f);
            const float err = __fdiv_rn(__fsub_rn(wi, __fmul_rn(q, br)), Hi[(size_t)i * ldh + i]);
            if (lane == o) {
                qrow[i] = q;
                erow[i] = err;
            }
            const float* hrow = Hi + (size_t)i * ldh;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (k > s || (k == s && lane > o)) {   // columns j > i (column i itself is never read again)
                    const int j = lane + 32 * k;
                    r[k] = __fsub_rn(r[k], __fmul_rn(err, hrow[j]));
                }
            }
        }
    }
}
