#include "../include/neonflux/gemm.h"
#include <algorithm>
#include <arm_neon.h>
#include <cstring>
#include <omp.h>
#include <vector>

namespace neonflux {
void gemm_ref(int M, int N, int K, const float *A, const float *B, float *C) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int p = 0; p < K; ++p) {
        sum += A[i * K + p] * B[p * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

#define MC 256
#define KC 128
#define NC 128

#define MR 8
#define NR 4

inline void transpose_4x4_neon(float32x4_t &r0, float32x4_t &r1, float32x4_t &r2, float32x4_t &r3) {
  float32x4_t t0 = vtrn1q_f32(r0, r1);
  float32x4_t t1 = vtrn2q_f32(r0, r1);
  float32x4_t t2 = vtrn1q_f32(r2, r3);
  float32x4_t t3 = vtrn2q_f32(r2, r3);

  float64x2_t d0 = vreinterpretq_f64_f32(t0);
  float64x2_t d1 = vreinterpretq_f64_f32(t1);
  float64x2_t d2 = vreinterpretq_f64_f32(t2);
  float64x2_t d3 = vreinterpretq_f64_f32(t3);

  r0 = vreinterpretq_f32_f64(vtrn1q_f64(d0, d2));
  r1 = vreinterpretq_f32_f64(vtrn1q_f64(d1, d3));
  r2 = vreinterpretq_f32_f64(vtrn2q_f64(d0, d2));
  r3 = vreinterpretq_f32_f64(vtrn2q_f64(d1, d3));
}

inline void pack_matrix_b_4(int k, const float *B, int ldb, int valid_cols, float *buffer) {
  for (int p = 0; p < k; ++p) {
    for (int j = 0; j < NR; ++j) {
      buffer[p * NR + j] = (j < valid_cols) ? B[p * ldb + j] : 0.0f;
    }
  }
}

inline void pack_matrix_a_8xk(int k, const float *A, int lda, int valid_rows, float *buffer) {
  if (valid_rows == MR) {
    int p = 0;

    for (; p + 3 < k; p += 4) {
      float32x4_t r0 = vld1q_f32(&A[0 * lda + p]);
      float32x4_t r1 = vld1q_f32(&A[1 * lda + p]);
      float32x4_t r2 = vld1q_f32(&A[2 * lda + p]);
      float32x4_t r3 = vld1q_f32(&A[3 * lda + p]);
      transpose_4x4_neon(r0, r1, r2, r3);

      float32x4_t r4 = vld1q_f32(&A[4 * lda + p]);
      float32x4_t r5 = vld1q_f32(&A[5 * lda + p]);
      float32x4_t r6 = vld1q_f32(&A[6 * lda + p]);
      float32x4_t r7 = vld1q_f32(&A[7 * lda + p]);
      transpose_4x4_neon(r4, r5, r6, r7);

      vst1q_f32(buffer + 0, r0);
      vst1q_f32(buffer + 4, r4);
      buffer += MR;

      vst1q_f32(buffer + 0, r1);
      vst1q_f32(buffer + 4, r5);
      buffer += MR;

      vst1q_f32(buffer + 0, r2);
      vst1q_f32(buffer + 4, r6);
      buffer += MR;

      vst1q_f32(buffer + 0, r3);
      vst1q_f32(buffer + 4, r7);
      buffer += MR;
    }

    for (; p < k; ++p) {
      for (int i = 0; i < MR; ++i) {
        *buffer++ = A[i * lda + p];
      }
    }
    return;
  }

  for (int p = 0; p < k; ++p) {
    for (int i = 0; i < MR; ++i) {
      *buffer++ = (i < valid_rows) ? A[i * lda + p] : 0.0f;
    }
  }
}

inline void kernel_8x4_accum(int k, const float *A_packed, const float *B_packed, float *C, int ldc) {
  float32x4_t c0 = vdupq_n_f32(0.0f);
  float32x4_t c1 = vdupq_n_f32(0.0f);
  float32x4_t c2 = vdupq_n_f32(0.0f);
  float32x4_t c3 = vdupq_n_f32(0.0f);
  float32x4_t c4 = vdupq_n_f32(0.0f);
  float32x4_t c5 = vdupq_n_f32(0.0f);
  float32x4_t c6 = vdupq_n_f32(0.0f);
  float32x4_t c7 = vdupq_n_f32(0.0f);

  const float *a_ptr = A_packed;
  const float *b_ptr = B_packed;

  for (int p = 0; p < k; ++p) {
    float32x4_t b = vld1q_f32(b_ptr);
    b_ptr += NR;

    float32x4_t a0 = vdupq_n_f32(a_ptr[0]);
    float32x4_t a1 = vdupq_n_f32(a_ptr[1]);
    float32x4_t a2 = vdupq_n_f32(a_ptr[2]);
    float32x4_t a3 = vdupq_n_f32(a_ptr[3]);
    float32x4_t a4 = vdupq_n_f32(a_ptr[4]);
    float32x4_t a5 = vdupq_n_f32(a_ptr[5]);
    float32x4_t a6 = vdupq_n_f32(a_ptr[6]);
    float32x4_t a7 = vdupq_n_f32(a_ptr[7]);
    a_ptr += MR;

    c0 = vmlaq_f32(c0, a0, b);
    c1 = vmlaq_f32(c1, a1, b);
    c2 = vmlaq_f32(c2, a2, b);
    c3 = vmlaq_f32(c3, a3, b);
    c4 = vmlaq_f32(c4, a4, b);
    c5 = vmlaq_f32(c5, a5, b);
    c6 = vmlaq_f32(c6, a6, b);
    c7 = vmlaq_f32(c7, a7, b);
  }

  float32x4_t old0 = vld1q_f32(C + 0 * ldc);
  float32x4_t old1 = vld1q_f32(C + 1 * ldc);
  float32x4_t old2 = vld1q_f32(C + 2 * ldc);
  float32x4_t old3 = vld1q_f32(C + 3 * ldc);
  float32x4_t old4 = vld1q_f32(C + 4 * ldc);
  float32x4_t old5 = vld1q_f32(C + 5 * ldc);
  float32x4_t old6 = vld1q_f32(C + 6 * ldc);
  float32x4_t old7 = vld1q_f32(C + 7 * ldc);

  vst1q_f32(C + 0 * ldc, vaddq_f32(old0, c0));
  vst1q_f32(C + 1 * ldc, vaddq_f32(old1, c1));
  vst1q_f32(C + 2 * ldc, vaddq_f32(old2, c2));
  vst1q_f32(C + 3 * ldc, vaddq_f32(old3, c3));
  vst1q_f32(C + 4 * ldc, vaddq_f32(old4, c4));
  vst1q_f32(C + 5 * ldc, vaddq_f32(old5, c5));
  vst1q_f32(C + 6 * ldc, vaddq_f32(old6, c6));
  vst1q_f32(C + 7 * ldc, vaddq_f32(old7, c7));
}

void gemm_optimized(int M, int N, int K, const float *A, const float *B, float *C) {
  std::fill(C, C + static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);

#pragma omp parallel for schedule(static)
  for (int j = 0; j < N; j += NC) {
    std::vector<float> packA(MC * KC);
    std::vector<float> packB(KC * NC);

    const int nc = std::min(NC, N - j);

    for (int k = 0; k < K; k += KC) {
      const int kc = std::min(KC, K - k);
      for (int jj = 0; jj < nc; jj += NR) {
        const int valid_cols = std::min(NR, nc - jj);
        const int b_panel_idx = jj / NR;
        float *b_dst = packB.data() + b_panel_idx * kc * NR;

        pack_matrix_b_4(kc, &B[k * N + (j + jj)], N, valid_cols, b_dst);
      }

      for (int i = 0; i < M; i += MC) {
        const int mc = std::min(MC, M - i);
        for (int ii = 0; ii < mc; ii += MR) {
          const int valid_rows = std::min(MR, mc - ii);
          const int a_panel_idx = ii / MR;
          float *a_dst = packA.data() + a_panel_idx * kc * MR;

          pack_matrix_a_8xk(kc, &A[(i + ii) * K + k], K, valid_rows, a_dst);
        }

        for (int jj = 0; jj < nc; jj += NR) {
          const int valid_cols = std::min(NR, nc - jj);
          const int b_panel_idx = jj / NR;
          const float *b_tile = packB.data() + b_panel_idx * kc * NR;
          for (int ii = 0; ii < mc; ii += MR) {
            const int valid_rows = std::min(MR, mc - ii);
            const int a_panel_idx = ii / MR;
            const float *a_tile = packA.data() + a_panel_idx * kc * MR;
            if (valid_rows == MR && valid_cols == NR) {
              kernel_8x4_accum(kc, a_tile, b_tile, &C[(i + ii) * N + (j + jj)], N);
            } else {
              float tempC[MR * NR];
              std::memset(tempC, 0, sizeof(tempC));
              kernel_8x4_accum(kc, a_tile, b_tile, tempC, NR);
              for (int r = 0; r < valid_rows; ++r) {
                for (int c = 0; c < valid_cols; ++c) {
                  C[(i + ii + r) * N + (j + jj + c)] += tempC[r * NR + c];
                }
              }
            }
          }
        }
      }
    }
  }
}
}