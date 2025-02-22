#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include <math.h>
#include <stdio.h>
#include <iostream>

using rocwmma::float16_t;
using rocwmma::float32_t;

#define  half_t half
#define  half2_t half2
#define rocwmma_half half
[[maybe_unused]]
constexpr uint32_t warp_size = 64;


#define FLOAT2(pointer) (reinterpret_cast<float2*>((void *)&(pointer))[0])

#define CHECK_CUDA_ERROR(val) checkCudaError((val), #val, __FILE__, __LINE__)
template <typename T>
void checkCudaError(T err, char const* const func, char const* const file,
        int const line)
{
    if (err != hipSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                << std::endl;
        std::cerr << hipGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLastCudaError(__FILE__, __LINE__)
void checkLastCudaError(char const* const file, int const line)
{
    hipError_t err{hipGetLastError()};
    if (err != hipSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                << std::endl;
        std::cerr << hipGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}



// __device__ half __hmax(half a, half b) {
//     return __hgt(a, b) ? a : b;
// }

__device__ half2 __hmax2(half2 a, half2 b) {
    half tmp1;
    half tmp2;
    tmp1 = __hmax(__low2half(a), __low2half(b));
    tmp2 = __hmax(__high2half(a), __high2half(b));
    return __halves2half2(tmp1, tmp2);
}

// load 1xwidth data, padding (height - 1) * width areas.
template <uint32_t height, uint32_t width, uint32_t n_warps>
__device__
void threadblock_load_q_1xhead_dim(
    const half_t* __restrict__ src,
    half_t* __restrict__ dst,
    uint32_t lds,
    uint32_t ldd,
    uint32_t seq_len_q
) {
    if (blockDim.y == 1 && width == 128) {
        constexpr uint32_t n_threads = warp_size * n_warps;
        constexpr uint32_t elements_per_storage = 2;
        const uint32_t lane_idx = threadIdx.x;
        const uint32_t warp_idx = threadIdx.y;

        const uint32_t storage_idx = (warp_idx * warp_size + lane_idx) * elements_per_storage;
        
        // const uint32_t row = storage_idx / width;
        const uint32_t col = storage_idx;
        for (int offset = 0; offset < seq_len_q; offset++) {
            *(float*)&dst[(offset) * ldd + col] = *(float*)&src[(offset) * lds + col];
        }
        #pragma unroll
        for (uint32_t offset = seq_len_q; offset < height; offset++) {
            dst[(offset) * ldd + col] = __float2half(0.0f);
            dst[(offset) * ldd + col + 1] = __float2half(0.0f);
        }
    } else if (blockDim.y == 1 && width == 64) {
        constexpr uint32_t n_threads = warp_size * n_warps;
        constexpr uint32_t elements_per_storage = 1;
        const uint32_t lane_idx = threadIdx.x;
        const uint32_t warp_idx = threadIdx.y;

        const uint32_t storage_idx = (warp_idx * warp_size + lane_idx) * elements_per_storage;
        
        // const uint32_t row = storage_idx / width;
        const uint32_t col = storage_idx;
        for (int offset = 0; offset < seq_len_q; offset++) {
            dst[(offset) * ldd + col] = src[(offset) * lds + col];
        }
        #pragma unroll
        for (uint32_t offset = seq_len_q; offset < height; offset++) {
            dst[(offset) * ldd + col] = __float2half(0.0f);
        }
    } else {
        constexpr uint32_t n_threads = warp_size * n_warps;
        uint32_t elements_per_storage = 1; // 8 half_t == 1x uint4 == 128 bit
        uint32_t rows_per_iter = n_threads * elements_per_storage / width;

        // static_assert(n_threads * elements_per_storage % width == 0);

        const uint32_t lane_idx = threadIdx.x;
        const uint32_t warp_idx = threadIdx.y;

        const uint32_t storage_idx = (warp_idx * warp_size + lane_idx) * elements_per_storage;

        // for (int elements_per_storage_index = 0; elements_per_storage_index < elements_per_storage; elements_per_storage_index++) {
        //     const uint32_t row = (storage_idx + elements_per_storage_index) / width;
        //     const uint32_t col = (storage_idx + elements_per_storage_index) % width;
        //     if (col < width) {
        //         if (row < seq_len_q) {
        //             dst[row * ldd + col] = src[row * lds + col];
        //         } else {
        //             dst[row * ldd + col] = __float2half(0.0f);
        //         }
        //     }
        // }

        const uint32_t row = storage_idx / width;
        const uint32_t col = storage_idx % width;

        #pragma unroll
        for (uint32_t offset = 0; offset < height; offset += rows_per_iter) {
            if (offset + row < seq_len_q) {
                for (int col_index = 0; col_index < elements_per_storage; col_index++) {
                    dst[(offset + row) * ldd + col] = src[(offset + row) * lds + col];
                }
            } else {
                for (int col_index = 0; col_index < elements_per_storage; col_index++) {
                    dst[(offset + row) * ldd + col] = __float2half(0.0f);
                }
            }
        }
    }
}

template <uint32_t height, uint32_t width, uint32_t n_warps>
__device__
void threadblock_divide_and_store_chunk(
    const half_t* __restrict__ numer,
    const half_t* __restrict__ denom,
    half_t* __restrict__ dst,
    uint32_t lds,
    uint32_t ldd,
    uint32_t seq_len_q
) {
    if (blockDim.y == 1 && width == 128) {
        constexpr uint32_t n_threads = warp_size * n_warps;
        uint32_t elements_per_storage = 2;
        // uint32_t rows_per_iter = n_threads * elements_per_storage / width;

        // static_assert(n_threads * elements_per_storage % width == 0);

        const uint32_t lane_idx = threadIdx.x;
        const uint32_t warp_idx = threadIdx.y;
        const uint32_t storage_idx = (warp_idx * warp_size + lane_idx) * elements_per_storage;
        
        // const uint32_t row = storage_idx / width;
        const uint32_t col = storage_idx;

        for (int offset = 0; offset < seq_len_q; offset++) {
            *(half2_t*)&dst[(offset) * ldd + col] = __h2div(*(half2_t*)&numer[(offset) * lds + col], __half2half2(denom[offset]));
        }
    } else if (blockDim.y == 1 && width == 64) {
        constexpr uint32_t n_threads = warp_size * n_warps;
        uint32_t elements_per_storage = 1;
        // uint32_t rows_per_iter = n_threads * elements_per_storage / width;

        // static_assert(n_threads * elements_per_storage % width == 0);

        const uint32_t lane_idx = threadIdx.x;
        const uint32_t warp_idx = threadIdx.y;
        const uint32_t storage_idx = (warp_idx * warp_size + lane_idx) * elements_per_storage;
        
        // const uint32_t row = storage_idx / width;
        const uint32_t col = storage_idx;

        for (int offset = 0; offset < seq_len_q; offset++) {
            dst[(offset) * ldd + col] = __hdiv(numer[(offset) * lds + col], denom[offset]);
        }
    } else {
        constexpr uint32_t n_threads = warp_size * n_warps;
        uint32_t elements_per_storage = (seq_len_q * width + n_threads - 1) / n_threads;
        uint32_t rows_per_iter = n_threads * elements_per_storage / width;

        // static_assert(n_threads * elements_per_storage % width == 0);

        const uint32_t lane_idx = threadIdx.x;
        const uint32_t warp_idx = threadIdx.y;
        const uint32_t storage_idx = (warp_idx * warp_size + lane_idx) * elements_per_storage;
        
        for (int elements_per_storage_index = 0; elements_per_storage_index < elements_per_storage; elements_per_storage_index++) {
            const uint32_t row = (storage_idx + elements_per_storage_index) / width;
            const uint32_t col = (storage_idx + elements_per_storage_index) % width;
            if (row < seq_len_q && col < width) {
                dst[row * ldd + col] = __hdiv(
                numer[row * lds + col],
                denom[row]);
            }
        }
        // for (int offset = 0; offset < seq_len_q; offset++) {
        //     if (offset + row < seq_len_q) {
        //         for (int i = 0; i < elements_per_storage; i++) {
        //             dst[(offset + row) * ldd + col + i] = __hdiv(
        //             numer[(offset + row) * lds + col + i],
        //             denom[offset + row]);
        //         }
        //     }
        // }
        // #pragma unroll
        // for (uint32_t offset = seq_len_q; offset < height; offset++) {
        //     dst[(offset) * ldd + col] = __float2half(0.0f);
        // }

        // if (col < width) {
        //     dst[col] = __hdiv(
        //         numer[col],
        //         denom[0]
        //     );
        // }
    }
    // #pragma unroll
    // for (uint32_t offset = 0; offset < height; offset += rows_per_iter) {
    //     if (offset + row < seq_len_q) {
    //         *(half2_t*)&dst[(offset + row) * ldd + col] = __h2div(
    //             *(half2_t*)&numer[(offset + row) * lds + col],
    //             __half2half2(denom[offset + row])
    //         );
    //         // printf("%f %f \n", __half2float(numer[(offset + row) * lds + col]), __half2float(denom[offset + row]));
    //         // dst[(offset + row) * ldd + col] = __hdiv(
    //         //     numer[(offset + row) * lds + col],
    //         //     denom[offset + row]
    //         // );

    //         // dst[(offset + row) * ldd + col] = numer[0];
    //     }
    // }
    // if (row < seq_len) {
    //     #pragma unroll
    //     for (uint32_t offset = 0; offset < height; offset += rows_per_iter) {
    //         *(half2_t*)&dst[(offset + row) * ldd + col] = __h2div(
    //             *(half2_t*)&numer[(offset + row) * lds + col],
    //             __half2half2(denom[offset + row])
    //         );
    //     }
    // }
}

template < uint32_t m, uint32_t n, uint32_t k, int n_warps, 
        bool transpose_b=false, bool alpha_is_one=true, bool preloaded_a_frags=false>
__device__
void threadblock_gemm_k_real(
    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, rocwmma_half, rocwmma::row_major> mat_a_frags[m / 16][k / 16],
    const half_t* __restrict__ mat_a,
    const half_t* __restrict__ mat_b,
    half_t* __restrict__ mat_c,
    uint32_t m_real,
    uint32_t n_real,
    uint32_t k_real,
    uint32_t lda,
    uint32_t ldb,
    uint32_t ldc,
    __half alpha = __float2half(1.0f)
) {
    constexpr uint32_t frag_cols_per_warp = (n + 16 * n_warps - 1) / (16 * n_warps); // num of columns processed by single warp (in terms of fragments)
    constexpr uint32_t frag_rows = m / 16; // num of fragments in rows in output matrix =1
    constexpr uint32_t frag_cols = n / 16; // num of fragments in cols in output matrix
    constexpr uint32_t frag_dims = k / 16; // size of common dimention of a and b matrices (in fragments)

    // static_assert(n_warps * frag_cols_per_warp == frag_cols);
    // static_assert(n % (n_warps * 16) == 0);

    using mat_b_order = std::conditional_t<transpose_b, rocwmma::col_major, rocwmma::row_major>;

    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, rocwmma_half, rocwmma::row_major> frag_a;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, rocwmma_half, mat_b_order> frag_b[frag_dims][frag_cols_per_warp];
    // rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, rocwmma_half> frag_acc;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> frag_acc;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, rocwmma_half> frag_c;

    const uint32_t warp_idx = threadIdx.y;

    // Load mat b to fragments distributed by cols between warps 
    // if n_warps is smaller than fragments' columns then one warp works with several columns
    #pragma unroll
    for (uint32_t frag_dim = 0; frag_dim < frag_dims; ++frag_dim) { // k_loop
        #pragma unroll
        for (uint32_t frag_col_offset = 0; frag_col_offset < frag_cols_per_warp; frag_col_offset++) { // n_loop
            const uint32_t frag_col = warp_idx * frag_cols_per_warp + frag_col_offset;
            if (16 * frag_col >= n_real || 16 * frag_dim >= k_real) continue;
            if (transpose_b) {
                int n_index = threadIdx.x / 16;
                int m_index = threadIdx.x % 16;
                int B_m_index = m_index;
                int B_n_index = n_index * 4;
                auto d_B = &mat_b[16 * (frag_dim + frag_col * ldb)];
                half reg_B[4] = {__float2half(0.0f)};
                // FLOAT2(reg_B[0]) = FLOAT2(d_B[B_m_index * ldb + B_n_index]);
                if (16 * frag_col + B_m_index < n_real) {
                    // FLOAT2(reg_B[0]) = FLOAT2(d_B[B_m_index * ldb + B_n_index]);
                    if (16 * frag_dim + B_n_index < k_real) reg_B[0] = d_B[B_m_index * ldb + B_n_index];
                    if (16 * frag_dim + B_n_index + 1 < k_real) reg_B[1] = d_B[B_m_index * ldb + B_n_index + 1];
                    if (16 * frag_dim + B_n_index + 2 < k_real) reg_B[2] = d_B[B_m_index * ldb + B_n_index + 2];
                    if (16 * frag_dim + B_n_index + 3 < k_real) reg_B[3] = d_B[B_m_index * ldb + B_n_index + 3];
                }
                frag_b[frag_dim][frag_col_offset].x[0] = reg_B[0];
                frag_b[frag_dim][frag_col_offset].x[1] = reg_B[1];
                frag_b[frag_dim][frag_col_offset].x[2] = reg_B[2];
                frag_b[frag_dim][frag_col_offset].x[3] = reg_B[3];

                // rocwmma::load_matrix_sync(frag_b[frag_dim][frag_col_offset],
                //                     &mat_b[16 * (frag_dim + frag_col * ldb)], ldb);
            } else {
                int n_index = threadIdx.x % 16;
                int m_index = threadIdx.x / 16;
                int B_m_index = m_index * 4;
                int B_n_index = n_index;
                int B_index = B_m_index * ldb + B_n_index;
                auto d_B = &mat_b[16 * (frag_dim * ldb + frag_col)];
                frag_b[frag_dim][frag_col_offset].x[0] = 16 * frag_dim + B_m_index < k_real && 16 * frag_col + B_n_index < n_real ? d_B[B_index] : __float2half(0.0f);
                B_m_index++;
                B_index += ldb;
                frag_b[frag_dim][frag_col_offset].x[1] = 16 * frag_dim + B_m_index < k_real && 16 * frag_col + B_n_index < n_real ? d_B[B_index] : __float2half(0.0f);
                B_m_index++;
                B_index += ldb;
                frag_b[frag_dim][frag_col_offset].x[2] = 16 * frag_dim + B_m_index < k_real && 16 * frag_col + B_n_index < n_real ? d_B[B_index] : __float2half(0.0f);
                B_m_index++;
                B_index += ldb;
                frag_b[frag_dim][frag_col_offset].x[3] = 16 * frag_dim + B_m_index < k_real && 16 * frag_col + B_n_index < n_real ? d_B[B_index] : __float2half(0.0f);

                // rocwmma::load_matrix_sync(frag_b[frag_dim][frag_col_offset],
                //                     &mat_b[16 * (frag_dim * ldb + frag_col)], ldb);
            }

            // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            //     auto d_B = &mat_b[16 * (frag_dim + frag_col * ldb)];
            //     if (!transpose_b) {
            //         d_B = &mat_b[16 * (frag_dim * ldb + frag_col)];
            //     }
            //     printf("frag_b: %f, %f, %f, %f \n", __half2float(frag_b[frag_dim][frag_col_offset].x[0]),  __half2float(frag_b[frag_dim][frag_col_offset].x[1]),  __half2float(frag_b[frag_dim][frag_col_offset].x[2]),  __half2float(frag_b[frag_dim][frag_col_offset].x[3]));
            //     printf("mat_b: \n");
            //     for (int i = 0; i < 16; i++) {
            //         for (int j = 0; j < 16; j++) {
            //             printf("%.1f ", __half2float(d_B[i * ldb + j]));
            //         }
            //         printf("\n");
            //     }
            //     printf("\n");
            // }
        }
    }

    // Iter trough rows
    #pragma unroll
    for (uint32_t frag_row = 0; frag_row < frag_rows; ++frag_row) {
        // Iter trough columns of single warp
        #pragma unroll
        for (uint32_t frag_col_offset = 0; frag_col_offset < frag_cols_per_warp; ++frag_col_offset) { // n loop
            const uint32_t frag_col = warp_idx * frag_cols_per_warp + frag_col_offset;
            if (16 * frag_col >= n_real || 16 * frag_row >= m_real) continue;
            // rocwmma::fill_fragment(frag_acc, __float2half(0.0f));
            rocwmma::fill_fragment(frag_acc, (0.0f));
            #pragma unroll
            for (uint32_t frag_dim = 0; frag_dim < frag_dims; ++frag_dim) { // k loop
                if (16 * frag_dim >= k_real) continue;
                if (preloaded_a_frags) {
                    rocwmma::mma_sync(frag_acc, mat_a_frags[frag_row][frag_dim], frag_b[frag_dim][frag_col_offset], frag_acc);
                } else {
                    rocwmma::load_matrix_sync(frag_a, &mat_a[16 * (frag_row * lda + frag_dim)], lda);
                    rocwmma::mma_sync(frag_acc, frag_a, frag_b[frag_dim][frag_col_offset], frag_acc);    
                }

                // __syncthreads();
                // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
                //     if (preloaded_a_frags) {
                //         printf("frag_a: %f, %f, %f, %f \n", __half2float(mat_a_frags[frag_row][frag_dim].x[0]),  __half2float(mat_a_frags[frag_row][frag_dim].x[1]),  __half2float(mat_a_frags[frag_row][frag_dim].x[2]),  __half2float(mat_a_frags[frag_row][frag_dim].x[3]));
                //     } else {
                //         printf("frag_a: %f, %f, %f, %f \n", __half2float(frag_a.x[0]),  __half2float(frag_a.x[1]),  __half2float(frag_a.x[2]),  __half2float(frag_a.x[3]));
                //     }
                //     printf("k_real: %f, %f, %f, %f \n", __half2float(frag_b[frag_dim][frag_col_offset].x[0]),  __half2float(frag_b[frag_dim][frag_col_offset].x[1]),  __half2float(frag_b[frag_dim][frag_col_offset].x[2]),  __half2float(frag_b[frag_dim][frag_col_offset].x[3]));
                //     printf("k_real: %f, %f, %f, %f \n", __half2float(frag_acc.x[0]),  __half2float(frag_acc.x[1]),  __half2float(frag_acc.x[2]),  __half2float(frag_acc.x[3]));
                // }
            }
            for(int t = 0; t < frag_acc.num_elements; t++) {
                frag_c.x[t] = __float2half(frag_acc.x[t]);
            }
            // multiply result by alpha
            if (!alpha_is_one) {
                for(int t = 0; t < frag_acc.num_elements; t++) {
                    // frag_acc.x[t] = frag_acc.x[t] * __half2float(alpha);
                    frag_c.x[t] = __hmul(frag_c.x[t], alpha);
                    // frag_acc.x[t] = __hmul(frag_acc.x[t], alpha);
                }
            }

            // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            //    printf("frag_a: %f, %f, %f, %f \n", __half2float(frag_a.x[0]),  __half2float(frag_a.x[1]),  __half2float(frag_a.x[2]),  __half2float(frag_a.x[3]));
            // //    printf("mat_a_frags: %f, %f, %f, %f \n", __half2float(mat_a_frags[0][0].x[0]),  __half2float(mat_a_frags[0][0].x[1]),  __half2float(mat_a_frags[0][0].x[2]),  __half2float(mat_a_frags[0][0].x[3]));
            //    printf("k_real: %f, %f, %f, %f \n", __half2float(frag_acc.x[0]),  __half2float(frag_acc.x[1]),  __half2float(frag_acc.x[2]),  __half2float(frag_acc.x[3]));
            // }
            rocwmma::store_matrix_sync(&mat_c[16 * (frag_row * ldc + frag_col)], frag_c, ldc, rocwmma::mem_row_major);

            // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            //     // printf("mat_c: \n");
            //     // for (int i = 0; i < 16; i++) {
            //     //     for (int j = 0; j < 16; j++) {
            //     //         printf("%.1f ", __half2float((&mat_c[16 * (frag_row * ldc + frag_col)])[i * ldc + j]));
            //     //     }
            //     //     printf("\n");
            //     // }
            //     // printf("\n");
            //     if (preloaded_a_frags) {
            //         printf("frag_a: %f, %f, %f, %f \n", __half2float(mat_a_frags[0][0].x[0]),  __half2float(mat_a_frags[0][0].x[1]),  __half2float(mat_a_frags[0][0].x[2]),  __half2float(mat_a_frags[0][0].x[3]));
            //     } else {
            //         printf("mat_a: \n");
            //         for (int i = 0; i < 16; i++) {
            //             for (int j = 0; j < 16; j++) {
            //                 printf("%.1f ", __half2float((&mat_a[16 * (frag_row * lda + 0)])[i * lda + j]));
            //             }
            //             printf("\n");
            //         }
            //     }
            //     printf("mat_b: \n");
            //     for (int i = 0; i < 16; i++) {
            //         for (int j = 0; j < 16; j++) {
            //             printf("%.1f ", __half2float((&mat_b[16 * (0 + frag_col * ldb)])[i * ldb + j]));
            //         }
            //         printf("\n");
            //     }
            //     printf("\n");
            //     printf("mat_c: \n");
            //     for (int i = 0; i < 16; i++) {
            //         for (int j = 0; j < 16; j++) {
            //             printf("%.1f ", __half2float((&mat_c[16 * (frag_row * ldc + frag_col)])[i * ldc + j]));
            //         }
            //         printf("\n");
            //     }
            //     printf("\n");
            //     printf("k_real: %f, %f, %f, %f \n", __half2float(frag_acc.x[0]),  __half2float(frag_acc.x[1]),  __half2float(frag_acc.x[2]),  __half2float(frag_acc.x[3]));
            // }
        }
    }
}
/*
* Calculates rowwise sums of matrix
* Uses additional auxillary matrix which should be of size (height, 16) to store fragmens with rowwise sums into shared memory
*/
template <uint32_t height, uint32_t width, uint32_t n_warps, bool copy_to_vec=true>
__device__
void threadblock_row_sum(
    const half_t* __restrict__ mat,
    half_t* __restrict__ vec,
    half_t* __restrict__ aux,
    uint32_t ldm,
    uint32_t ldm_aux
) {
    constexpr uint32_t frag_rows = height / 16;
    constexpr uint32_t frag_cols = width / 16;

    // static_assert(frag_rows % n_warps == 0);        // can distribute rows between warps (each warp calculates rowwise sum in one row of fragments)
    // static_assert(height <= n_warps * warp_size);   // can copy results to vec in single iteration

    const uint32_t warp_idx = threadIdx.y;
    const uint32_t lane_idx = threadIdx.x;
    const uint32_t storage_idx = (warp_idx * warp_size + lane_idx) * 2;

    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, rocwmma_half, rocwmma::row_major> frag_a;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, rocwmma_half, rocwmma::col_major> frag_b;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, rocwmma_half> frag_acc;
    // rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> frag_acc;
    // rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, rocwmma_half> frag_c;

    rocwmma::fill_fragment(frag_b, __float2half(1.0f));

    #pragma unroll
    for (uint32_t frag_row_offset = 0; frag_row_offset < frag_rows; frag_row_offset += n_warps) {
        const uint32_t frag_row = frag_row_offset + warp_idx;
        rocwmma::fill_fragment(frag_acc, __float2half(0.0f));
        // rocwmma::fill_fragment(frag_acc, (0.0f));

        #pragma unroll
        for(uint32_t frag_col = 0; frag_col < frag_cols; ++frag_col) {
            rocwmma::load_matrix_sync(frag_a, &mat[16 * (frag_row * ldm + frag_col)], ldm);
            rocwmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);
        }
        
        // for(int t = 0; t < frag_acc.num_elements; t++) {
        //     frag_c.x[t] = __float2half(frag_acc.x[t]);
        // }
        // Store in transposed mem to obtain all row-wise sums in coaleced vector in aux memory
        rocwmma::store_matrix_sync(&aux[16 * frag_row], frag_acc, ldm_aux, rocwmma::mem_col_major);
    }
    
    __syncthreads();

    if (copy_to_vec) {

        if (warp_idx * warp_size + lane_idx < height) {
            vec[warp_idx * warp_size + lane_idx] = aux[warp_idx * warp_size + lane_idx];
        }
    }
}
template <uint32_t height, uint32_t width, uint32_t n_warps, bool copy_to_vec=true>
__device__
void threadblock_row_sum(
    const half_t* __restrict__ mat,
    half_t* __restrict__ vec,
    half_t* __restrict__ aux,
    uint32_t ldm,
    uint32_t ldm_aux,
    uint32_t seq_len_q,
    uint32_t seq_len_k
) {
    constexpr uint32_t frag_rows = height / 16;
    constexpr uint32_t frag_cols = width / 16;

    // static_assert(frag_rows % n_warps == 0);        // can distribute rows between warps (each warp calculates rowwise sum in one row of fragments)
    // static_assert(height <= n_warps * warp_size);   // can copy results to vec in single iteration

    const uint32_t warp_idx = threadIdx.y;
    const uint32_t lane_idx = threadIdx.x;
    const uint32_t storage_idx = (warp_idx * warp_size + lane_idx) * 2;

    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, rocwmma_half, rocwmma::row_major> frag_a;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, rocwmma_half, rocwmma::col_major> frag_b;
    // rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, rocwmma_half> frag_acc;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> frag_acc;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, rocwmma_half> frag_c;

    rocwmma::fill_fragment(frag_b, __float2half(1.0f));

    #pragma unroll
    for (uint32_t frag_row_offset = 0; frag_row_offset < frag_rows; frag_row_offset += n_warps) {
        const uint32_t frag_row = frag_row_offset + warp_idx;
        if (frag_row * 16 >= seq_len_q) continue;
        // rocwmma::fill_fragment(frag_acc, __float2half(0.0f));
        rocwmma::fill_fragment(frag_acc, (0.0f));

        #pragma unroll
        for(uint32_t frag_col = 0; frag_col < frag_cols; ++frag_col) {
            if (frag_col * 16 >= seq_len_k) continue;
            rocwmma::load_matrix_sync(frag_a, &mat[16 * (frag_row * ldm + frag_col)], ldm);
            rocwmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);
        }
        
        for(int t = 0; t < frag_acc.num_elements; t++) {
            frag_c.x[t] = __float2half(frag_acc.x[t]);
        }
        // Store in transposed mem to obtain all row-wise sums in coaleced vector in aux memory
        rocwmma::store_matrix_sync(&aux[16 * frag_row], frag_c, ldm_aux, rocwmma::mem_col_major);
    }
    
    __syncthreads();

    if (copy_to_vec) {

        if (warp_idx * warp_size + lane_idx < height) {
            vec[warp_idx * warp_size + lane_idx] = aux[warp_idx * warp_size + lane_idx];
        }
    }
}


/*
* Calculates A = A + B elementwise
*/
template <uint32_t height, uint32_t width, uint32_t n_warps>
__device__ 
void threadblock_ewise_sum(
    half_t* __restrict__ mat_a,
    const half_t* __restrict__ mat_b,
    uint32_t lda,
    uint32_t ldb,
    uint32_t seq_len_q,
    uint32_t seq_len_k
) {
    constexpr uint32_t elements_per_storage = 2;
    constexpr uint32_t n_threads = warp_size * n_warps;
    constexpr uint32_t rows_per_iter = n_threads * elements_per_storage / width;

    // static_assert(n_threads * elements_per_storage % width == 0);

    const uint32_t lane_idx = threadIdx.x;
    const uint32_t warp_idx = threadIdx.y;

    const uint32_t storage_idx = (warp_idx * warp_size + lane_idx) * elements_per_storage;

    const uint32_t row = storage_idx / width;
    const uint32_t col = storage_idx % width;

    #pragma unroll
    for (uint32_t offset = 0; offset < height; offset += rows_per_iter) {
        if (offset + row < seq_len_q) {
            if (col + 1 < seq_len_k) {
                const uint32_t idx_a = (offset + row) * lda + col;
                const uint32_t idx_b = (offset + row) * ldb + col;

                *(half2_t*)&mat_a[idx_a] = __hadd2(*(half2_t*)&mat_a[idx_a], *(half2_t*)&mat_b[idx_b]);
            } else if (col < seq_len_k) {
                const uint32_t idx_a = (offset + row) * lda + col;
                const uint32_t idx_b = (offset + row) * ldb + col;

                mat_a[idx_a] = __hadd(mat_a[idx_a], mat_b[idx_b]);
            }
        }
    }
}

/*
* Calculates rowwise maximum of matrix and stores it to given vector
*/
template <uint32_t height, uint32_t width, uint32_t n_warps>
__device__
void threadblock_row_max(
    const half_t* __restrict__ mat,
    half_t* __restrict__ vec,
    half_t* __restrict__ aux,
    uint32_t ldm,
    uint32_t ldm_aux,
    uint32_t seq_len_q,
    uint32_t seq_len_k
) {
    constexpr uint32_t elements_per_storage = 2;
    constexpr uint32_t n_threads = warp_size * n_warps;
    constexpr uint32_t threads_per_row = n_threads / height;
    constexpr uint32_t storage_per_thread = width * height / (n_threads * elements_per_storage);

    // static_assert(n_threads % height == 0);                               // can equally distribute threads between rows
    // static_assert(width % (threads_per_row * elements_per_storage) == 0); // can distribute elements between all threads in the same row

    const uint32_t lane_idx = threadIdx.x;
    const uint32_t warp_idx = threadIdx.y;

    const uint32_t thread_idx = warp_idx * warp_size + lane_idx;
    const uint32_t storage_idx = thread_idx * storage_per_thread * elements_per_storage;

    const uint32_t row = storage_idx / width;
    const uint32_t col = storage_idx % width;
    const uint32_t col_aux = col / storage_per_thread;
    // if (row >= seq_len_q || col >= seq_len_k) return;
    // half2_t threadwise_val = *(half2_t*)&mat[row * ldm + col];

    // #pragma unroll
    // for (uint32_t i = 1; i < storage_per_thread; ++i) {

    //     if (row < seq_len_q && col + i * elements_per_storage + 1 < seq_len_k) {
    //         threadwise_val = __hmax2(threadwise_val, *(half2_t*)&mat[row * ldm + col + i * elements_per_storage]);
    //     } else if (row < seq_len_q && col + i * elements_per_storage < seq_len_k) {
    //         half_t mid_half = __hmax(__high2half(threadwise_val), mat[row * ldm + col + i * elements_per_storage]);
    //         threadwise_val = __halves2half2(__low2half(threadwise_val), mid_half);
    //     }
    // }

    // if (threads_per_row == 1) {
    //     vec[row] = __hmax(__high2half(threadwise_val), __low2half(threadwise_val));
    //     return;
    // }

    // *(half2_t*)&aux[row * ldm_aux + col_aux] = threadwise_val;
    
    // __syncthreads();
    
    if (thread_idx < seq_len_q) {
        half max_val = mat[thread_idx * ldm];
        // threadwise_val = *(half2_t*)&aux[thread_idx * ldm_aux];
        // const uint32_t thread_per_real =  ((seq_len_k + storage_per_thread - 1) / storage_per_thread + elements_per_storage - 1) / (elements_per_storage);
        #pragma unroll
        for (uint32_t i = 1; i < seq_len_k; ++i) {

            max_val = __hmax(max_val, mat[thread_idx * ldm + i]);
            // threadwise_val = __hmax2(threadwise_val, *(half2_t*)&aux[thread_idx * ldm_aux + i * elements_per_storage]);
        }

        vec[thread_idx] = max_val;
        // vec[thread_idx] = __hmax(__high2half(threadwise_val), __low2half(threadwise_val));

        // if (threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     // printf("mat: \n");
        //     // for (int i = 0; i < seq_len_q; i++) {
        //     //     for (int j = 0; j < seq_len_k; j++) {
        //     //         printf("%.1f ", __half2float(mat[i * ldm + j]));
        //     //     }
        //     //     printf("\n");
        //     // }
        //     printf("\n");printf("%.1f \n", __half2float(max_val));
        //     printf("%d %.1f \n", thread_idx, __half2float(vec[thread_idx]));
        //     printf("\n");
        // }
        
    }

    // __syncthreads();

    // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     printf("mat: \n");
    //     for (int i = 0; i < seq_len_q; i++) {
    //         for (int j = 0; j < seq_len_k; j++) {
    //             printf("%.1f ", __half2float(mat[i * ldm + j]));
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    //     printf("\n");printf("%.1f \n", __half2float(max_val));
    //     printf("%.1f \n", __half2float(vec[thread_idx]));
    //     printf("\n");
    // }

    // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {

    //     printf("aux: \n");
    //     for (int i = 0; i < height; i++) {
    //         for (int j = 0; j < width; j++) {
    //             printf("%.1f ", __half2float(aux[i * ldm_aux + j]));
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {

    //     printf("vec: \n");
    //     for (int i = 0; i < height; i++) {
    //         printf("%.1f ", __half2float(vec[i]));
    //     }
    //     printf("\n");
    // }
}

/*
* Calculates A = exp(A - b[:, None]), where A - matrix, b - column-vector
* Used for stable exp calculation in softmax
*/
template <uint32_t height, uint32_t width, uint32_t n_warps>
__device__
void threadblock_row_broadcast_diff_and_exp(
    half_t* __restrict__ mat,
    const half_t* __restrict__ vec,
    uint32_t ldm,
    uint32_t seq_len_q,
    uint32_t seq_len_k
) {
    constexpr uint32_t elements_per_storage = 2;
    constexpr uint32_t n_threads = warp_size * n_warps;
    constexpr uint32_t rows_per_iter = n_threads * elements_per_storage / width;

    // static_assert(n_threads * elements_per_storage % width == 0);

    const uint32_t lane_idx = threadIdx.x;
    const uint32_t warp_idx = threadIdx.y;

    const uint32_t storage_idx = (warp_idx * warp_size + lane_idx) * elements_per_storage;

    const uint32_t row = storage_idx / width;
    const uint32_t col = storage_idx % width;

    #pragma unroll
    for (uint32_t offset = 0; offset < height; offset += rows_per_iter) {
        if ((offset + row) < seq_len_q) {
            const uint32_t idx = (offset + row) * ldm + col;
            if (col + 1 < seq_len_k) {
                // mat[idx] = __float2half(expf(__half2float(__hsub(mat[idx], vec[offset + row]))));
                // mat[idx + 1] = __float2half(expf(__half2float(__hsub(mat[idx + 1], vec[offset + row]))));
                *(half2_t*)&mat[idx] = h2exp(__hsub2(*(half2_t*)&mat[idx], __half2half2(vec[offset + row])));
                // mat[idx] = hexp(__hsub(mat[idx], vec[offset + row]));
            } else if (col < seq_len_k) {
                // mat[idx] = __float2half(expf(__half2float(__hsub(mat[idx], vec[offset + row]))));
                mat[idx] = hexp(__hsub(mat[idx], vec[offset + row]));
            } else {
                mat[idx] = __float2half(0.0f);
                mat[idx + 1] = __float2half(0.0f);
            }

        }
        // if ((offset + row) < seq_len_q && col + 1 < seq_len_k) {
        //     const uint32_t idx = (offset + row) * ldm + col;
        //     // mat[idx] = __float2half(expf(__half2float(__hsub(mat[idx], vec[offset + row]))));
        //     // mat[idx + 1] = __float2half(expf(__half2float(__hsub(mat[idx + 1], vec[offset + row]))));
        //     *(half2_t*)&mat[idx] = h2exp(__hsub2(*(half2_t*)&mat[idx], __half2half2(vec[offset + row])));
        //     // mat[idx] = hexp(__hsub(mat[idx], vec[offset + row]));
        // } else if ((offset + row) < seq_len_q && col < seq_len_k) {
        //     const uint32_t idx = (offset + row) * ldm + col;
        //     // mat[idx] = __float2half(expf(__half2float(__hsub(mat[idx], vec[offset + row]))));
        //     mat[idx] = hexp(__hsub(mat[idx], vec[offset + row]));
        // }
    }
}

/*
* Calculates: 
*      max_a' = max(max_a, max_b)
*      numer_a' = numer_a * exp(max_a - max_a') + numer_b * exp(max_b - max_a')
*      denom_a' = denom_a * exp(max_a - max_a') + denom_b * exp(max_b - max_a')
* Which is casual sum: 
*      numer_A = numer_A + numer_B 
*      denom_a = denom_a + denom_b
* but all numers represented as x' = exp(max_x) * x and exponentials are not caclulated due to computation errors
*/
// chunk_size, head_dim, n_warps
template <uint32_t height, uint32_t width, uint32_t n_warps>
__device__
void threadblock_aggregate_softmax(
    half_t* __restrict__ numer_a,
    half_t* __restrict__ denom_a,
    half_t* __restrict__ max_a,
    const half_t* __restrict__ numer_b,
    const half_t* __restrict__ denom_b,
    half_t* __restrict__ max_b,
    uint32_t lda,
    uint32_t ldb,
    half_t* __restrict__ aux,
    uint32_t seq_len_q
) {
    constexpr uint32_t elements_per_storage = 2;
    constexpr uint32_t rows_per_iter = warp_size * n_warps * elements_per_storage / width;

    const uint32_t lane_idx = threadIdx.x;
    const uint32_t warp_idx = threadIdx.y;

    const uint32_t storage_idx = (warp_idx * warp_size + lane_idx) * elements_per_storage;

    if (storage_idx + 1 < seq_len_q) {
        half2_t* __restrict__ a2_ptr = (half2_t*)&max_a[storage_idx];
        half2_t* __restrict__ b2_ptr = (half2_t*)&max_b[storage_idx];
        half2_t* __restrict__ max_ab2_ptr = (half2_t*)&aux[storage_idx];
        half2_t mid_value = *a2_ptr;
        *max_ab2_ptr = __hmax2(*a2_ptr, *b2_ptr);
        *a2_ptr = h2exp(__hsub2(*a2_ptr, *max_ab2_ptr));
        *b2_ptr = h2exp(__hsub2(*b2_ptr, *max_ab2_ptr));
        
        *(half2_t*)&denom_a[storage_idx] = __hadd2(__hmul2(*a2_ptr, *(half2_t*)&denom_a[storage_idx]),
                                                __hmul2(*b2_ptr, *(half2_t*)&denom_b[storage_idx]));
    } else if (storage_idx < seq_len_q) {
        half_t* __restrict__ a_ptr = &max_a[storage_idx];
        half_t* __restrict__ b_ptr = &max_b[storage_idx];
        half_t* __restrict__ max_ab_ptr = &aux[storage_idx];
        half_t mid_value = *a_ptr;
        *max_ab_ptr = __hmax(*a_ptr, *b_ptr);
        *a_ptr = hexp(__hsub(*a_ptr, *max_ab_ptr));
        *b_ptr = hexp(__hsub(*b_ptr, *max_ab_ptr));
        
        denom_a[storage_idx] = __hadd(__hmul(*a_ptr, denom_a[storage_idx]),
                                                __hmul(*b_ptr, denom_b[storage_idx]));
    }

    __syncthreads();

    const uint32_t row = storage_idx / width;
    const uint32_t col = storage_idx % width;

    #pragma unroll
    for (uint32_t offset = 0; offset < height; offset += rows_per_iter) {
        const uint32_t idx_a = (offset + row) * lda + col;
        const uint32_t idx_b = (offset + row) * ldb + col;

        if (offset + row < seq_len_q) {
            *(half2_t*)&numer_a[idx_a] = __hadd2(__hmul2(__half2half2(max_a[offset + row]), *(half2_t*)&numer_a[idx_a]),
                                            __hmul2(__half2half2(max_b[offset + row]), *(half2_t*)&numer_b[idx_b]));
        }
    }

    __syncthreads();
    
    if (storage_idx < seq_len_q) {
        *(half2_t*)&max_a[storage_idx] = *(half2_t*)&aux[storage_idx];
    }
}

/*
* Usage: distribute_shared_mem<T, size_a, size_b, size_c>(shmem, &a, &b, &c);
* Where shmem is pointer to shared memory, and a, b, c are pointers to arrays of corresponding sizes
*/
template <typename T, int size>
__device__
inline void distribute_shared_mem(T* mem_ptr, T** array_ptr) {
    *array_ptr = mem_ptr;
}

template <typename T, int size, int... sizes, typename... Args>
__device__
inline void distribute_shared_mem(T* mem_ptr, T** array_ptr, Args... array_ptrs) {
    *array_ptr = mem_ptr;
    distribute_shared_mem<T, sizes...>(mem_ptr + size, array_ptrs...);
}

template <uint32_t height, uint32_t width>
__device__
void print_matrix(half_t* mat, uint32_t ldm) {
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (uint32_t i = 0; i < height; ++i) {
            for (uint32_t j = 0; j < width; ++j) {
                printf("%8.4f ", __half2float(mat[i * ldm + j]));
            }
            printf("\n");
        }
        printf("\n\n");
    }

    __syncthreads();
}



/*
chunk_size = 64 (or 128?)
head_size = 128

K = L / chunk_size
N_heads = F / head_size

queries     (batch_size, seq_len, num_features)
keys        -//-
values      -//-
attn_mask   (batch_size, seq_len, seq_len)

                            | elements  -> chunks (chunk_size, head_size)
S = softmax(Q @ K.T / sqrt(d)) | (B, L, L) -> (B, K, K)
Y = S @ V                      | (B, L, F) -> (B, K, N_heads)

for each batch = 1 .. n_batches:        // can be parallelized, gridDim.x
    for each head = 1 .. n_heads:       // can be parallelized, gridDim.y
        for each q_row = 1 .. K:        // can be parallelized, gridDim.z

            query_chunk = Q[q_row, head]    // Load query chunk to shmem

            for each kv_row = 1 .. K: // done seuqential inside threadblock
                // calculate score:
                // score[q_row, kv_row] = Q[q_row, head] @ K[kv_row, head].T / sqrt(head_dim)
                score_chunk = query_chunk @ K[kv_row, head].T / sqrt(head_dim)

                // score_max[q_row] = max_row(score[q_row, kv_row])
                score_max = max(score_chunk, dim=1)

                // score[q_row, kv_row] = exp(score[q_row, kv_row] - score_max[q_row])
                score_chunk = exp(score_chunk - score_max.view(chunk_size, 1))
                
                numer_local = score_chunk @ V[kv_row, head]
                denom_local = sum(score_chunk, dim=1)

                tmp_max = maximum(score_max, score_max_old)

                numer = exp(score_max - tmp_max) * numer + exp(score_max_local - tmp_max) * numer_local
                denom = exp(score_max - tmp_max) * denom + exp(score_max_local - tmp_max) * denom_local
                score_max = tmp_max

            // Y[q_row, head] = exp(logits_numer[q_row, head] - logits_denom[q_row, head])
            Y[q_row, head] = exp(logits_numer - logits_denom.view(chunk_size, 1))
*/
template <
    uint32_t head_dim,
    uint32_t chunk_size,
    uint32_t n_warps
>
__global__
__launch_bounds__(64) void attention_kernel_64(
    uint32_t batch_size,
    uint32_t seq_len_q,
    uint32_t seq_len_k,
    uint32_t num_features,
    const half_t* __restrict__ queries,
    const half_t* __restrict__ keys,
    const half_t* __restrict__ values,
    const half_t* __restrict__ mask,
    half_t* __restrict__ output
) {

    constexpr uint32_t mat_skew = 8;
    // constexpr uint32_t common_ldm = head_dim + mat_skew;
    constexpr uint32_t reduce_max_ldm = 2 * warp_size * n_warps / chunk_size;

    constexpr uint32_t chunk_frags = chunk_size / 16;
    constexpr uint32_t head_frags = head_dim / 16;

    constexpr uint32_t max_size = chunk_size < head_dim ? head_dim : chunk_size;

    // // static_assert(chunk_size <= head_dim);
    // static_assert(chunk_size <= warp_size * n_warps);                     // Column operations should be done in one iter (or less)
    // static_assert(chunk_size * head_dim % (warp_size * n_warps) == 0);    // Matrix operations should be done in several iters
    // static_assert(chunk_size * chunk_size % (warp_size * n_warps) == 0);
    // static_assert(reduce_max_ldm <= 16);

    half_t* queries_chunk;      // matrix (chunk_size, head_dim)
    half_t* scores_chunk;       // matrix (chunk_size, chunk_size)
    half_t* numer_local;        // matrix (chunk_size, head_dim)
    half_t* numer;              // matrix (chunk_size, head_dim)
    half_t* aux_mem;            // matrix (chunk_size, 16)

    constexpr uint32_t numer_ldm = head_dim + mat_skew;
    constexpr uint32_t queries_chunk_ldm = numer_ldm;
    constexpr uint32_t numer_local_ldm = max_size + mat_skew;
    constexpr uint32_t scores_chunk_ldm = numer_local_ldm;
    constexpr uint32_t reduce_sum_ldm = chunk_size + mat_skew;

    half_t* scores_max_local;   // vector (chunk_size,)
    half_t* denom_local;        // vector (chunk_size,)
    half_t* denom;              // vector (chunk_size,)
    half_t* scores_max;         // vector (chunk_size,)

    extern __shared__ half_t shmem[];

    distribute_shared_mem<half_t,
        chunk_size * numer_ldm,                      // numer / queries_chunk
        (chunk_size + 16) * numer_local_ldm,         // numer_local / scores_chunk
        16 * reduce_sum_ldm,                         // aux_mem
        chunk_size,                                  // scores_max_local
        chunk_size,                                  // denom
        chunk_size,                                  // denom_local
        chunk_size                                   // scores_max
    >(
        shmem,
        &numer,
        &numer_local,
        &aux_mem,
        &scores_max_local,
        &denom,
        &denom_local,
        &scores_max
    );

    queries_chunk = numer;
    scores_chunk = &numer_local[16 * numer_local_ldm];
    
    // const uint32_t chunk_size_q = 16;
    // const uint32_t chunk_size_k = chunk_size;
    const uint32_t num_chanks_k = (seq_len_k + chunk_size - 1) / chunk_size;
    const uint32_t chunk_size_q_remain = seq_len_q % chunk_size;
    const uint32_t chunk_size_k_remain = seq_len_k % chunk_size;

    const uint32_t batch_idx = blockIdx.x;
    const uint32_t q_row_chunk = blockIdx.y;
    const uint32_t head_col_chunk = blockIdx.z;
    // seq_len of  q, k and v may not be the same.
    const uint32_t batch_offset_q = batch_idx * seq_len_q * num_features;
    const uint32_t batch_offset_k = batch_idx * seq_len_k * num_features;
    const uint32_t batch_offset_mask = batch_idx * seq_len_q * seq_len_k;

    // row / col in the matrix of fixed batch_idx (seq_len, num_features)
    const uint32_t q_row = q_row_chunk * chunk_size;
    const uint32_t head_col = head_col_chunk * head_dim;
    const uint32_t chunk_size_q_real = (q_row_chunk == gridDim.y - 1 && chunk_size_q_remain > 0) ? chunk_size_q_remain : chunk_size;

    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, rocwmma_half, rocwmma::row_major> query_frags[chunk_frags][head_frags];
    // batch_size, sequence_len, num_heads, head_dim
    // Load query chunk into shared memory
    threadblock_load_q_1xhead_dim<chunk_size, head_dim, n_warps>(
        /*src = */ &queries[batch_offset_q + q_row * num_features + head_col],
        /*dst = */ queries_chunk,
        /*lds = */ num_features,
        /*ldd = */ queries_chunk_ldm,
        /*seq_len_q*/chunk_size_q_real
    );

    // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     printf("queries_chunk: \n");
    //     for (int i = 0; i < chunk_size; i++) {
    //         for (int j = 0; j < head_dim; j++) {
    //             printf("%.1f ", __half2float(queries_chunk[i * queries_chunk_ldm + j]));
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    __syncthreads();

    // Move query chunk into warps' fragments, query_chunk is freed here and can be used later as numer
    #pragma unroll
    for (uint32_t row_frag = 0; row_frag < chunk_frags; ++row_frag) {
        #pragma unroll
        for (uint32_t col_frag = 0; col_frag < head_frags; ++col_frag) {
            rocwmma::load_matrix_sync(query_frags[row_frag][col_frag], &queries_chunk[16 * (row_frag * queries_chunk_ldm + col_frag)], queries_chunk_ldm);
        }
    }

    #pragma unroll(1)
    for (uint32_t kv_row_chunk = 0; kv_row_chunk < num_chanks_k; kv_row_chunk++) {
        const uint32_t kv_row = kv_row_chunk * chunk_size;
        uint32_t chunk_size_k_real = chunk_size;
        if (kv_row_chunk == num_chanks_k - 1 && chunk_size_k_remain > 0) {
            chunk_size_k_real = chunk_size_k_remain;
        }
        // scores_chunk = queries_chunk @ keys_chunk.T / sqrt(head_dim)
        threadblock_gemm_k_real< /*m=*/chunk_size, /*n=*/chunk_size, /*k=*/head_dim, n_warps,
                        /*transpose_b=*/true, /*alpha_is_one=*/false, /*preloaded_a_frags=*/true >(
            /*mat_a_frags=*/ query_frags,
            /*mat_a =*/ nullptr,
            /*mat_b =*/ &keys[batch_offset_k + kv_row * num_features + head_col],
            /*mat_c =*/ scores_chunk,
                      chunk_size_q_real,
                      chunk_size_k_real,
                      head_dim,
            /*lda =*/ 0,
            /*ldb =*/ num_features,
            /*ldc =*/ scores_chunk_ldm,
            /*alpha =*/ __float2half(rsqrtf(head_dim))
        );
        // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("scores_chunk: \n");
        //     for (int i = 0; i < chunk_size; i++) {
        //         for (int j = 0; j < chunk_size; j++) {
        //             printf("%.1f ", __half2float(scores_chunk[i * queries_chunk_ldm + j]));
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }

        __syncthreads();


        if (kv_row_chunk == 0) {
            // First iter: write directly to numer, denom, scores_max, no aggregation

            if (mask != nullptr) {
                threadblock_ewise_sum<chunk_size, chunk_size, n_warps>(
                    /*mat_a =*/ scores_chunk,
                    /*mat_b =*/ &mask[kv_row],
                    /*lda =*/ scores_chunk_ldm,
                    seq_len_k,
                    chunk_size_q_real,
                    chunk_size_k_real
                );
            }
        }
        if (kv_row_chunk == 0) {
            // scores_max = max(scores_chunk, dim=1)
            threadblock_row_max<chunk_size, chunk_size, n_warps>(
                /*matrix =*/ scores_chunk,
                /*column =*/ scores_max,
                /*aux_memory =*/ aux_mem,
                /*ldm =*/ scores_chunk_ldm,
                /*ldm_aux =*/ reduce_max_ldm,
                chunk_size_q_real,
                chunk_size_k_real
            );
        }
        // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("scores_max: \n");
        //     for (int i = 0; i < chunk_size; i++) {
        //         printf("%.1f ", __half2float(scores_max[i]));
        //     }
        //     printf("\n");
        // }

        __syncthreads();
        
        if (kv_row_chunk == 0) {
            // scores_chunk = exp(scores_chunk - scores_max[:, None])
            threadblock_row_broadcast_diff_and_exp<chunk_size, chunk_size, n_warps>(
                /*matrix =*/ scores_chunk,
                /*column =*/ scores_max,
                /*ldm =*/ scores_chunk_ldm,
                chunk_size_q_real,
                chunk_size_k_real
            );
        }
        // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("diff_and_exp: \n");
        //     for (int i = 0; i < chunk_size; i++) {
        //         for (int j = 0; j < chunk_size; j++) {
        //             printf("%.1f ", __half2float(scores_chunk[i * queries_chunk_ldm + j]));
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }

        __syncthreads();
        
        if (kv_row_chunk == 0) {
            // denom = scores_chunk.sum(dim=1)
            threadblock_row_sum<chunk_size, chunk_size, n_warps>(
                /*mat = */ scores_chunk,
                /*vec = */ denom,
                /*aux_memory = */ aux_mem,
                /*ldm = */ scores_chunk_ldm,
                /*ldm_aux = */ reduce_sum_ldm
            );
        }
        
        __syncthreads();
        
        if (kv_row_chunk == 0) {
            // numer = scores_chunk @ values_chunk
            threadblock_gemm_k_real< /*m=*/chunk_size, /*n=*/head_dim, /*k=*/chunk_size, n_warps, 
                            /*transpose_b=*/false, /*alpha_is_one=*/true, /*preloaded_a_frags=*/false >(
                /*mat_a_frags = */ nullptr,
                /*mat_a = */ scores_chunk,
                /*mat_b = */ &values[batch_offset_k + head_col],
                /*mat_c = */ numer,
                /*m_real = */ chunk_size_q_real,
                /*n_real = */ head_dim,
                /*k_real = */ chunk_size_k_real,
                /*lda = */ scores_chunk_ldm,
                /*ldb = */ num_features,
                /*ldc = */ numer_ldm
            );

        }
        // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {

        //     printf("numer: \n");
        //     for (int i = 0; i < chunk_size; i++) {
        //         for (int j = 0; j < head_dim; j++) {
        //             printf("%.1f ", __half2float(numer[i * queries_chunk_ldm + j]));
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }

        // Second and further iterations, write to temprorary numer_local, denom_local, scores_max_local, then aggregate with prev values

        // scores_max = max(scores_chunk, dim=1)
        if (kv_row_chunk != 0) {
            threadblock_row_max<chunk_size, chunk_size, n_warps>(
                /*matrix =*/ scores_chunk,
                /*column =*/ scores_max_local,
                /*aux_memory =*/ aux_mem,
                /*ldm =*/ scores_chunk_ldm,
                /*ldm_aux =*/ reduce_max_ldm,
                chunk_size_q_real,
                chunk_size_k_real
            );
        }

        __syncthreads();

        if (kv_row_chunk != 0) {
            // scores_chunk = exp(scores_chunk - scores_max[:, None])
            threadblock_row_broadcast_diff_and_exp<chunk_size, chunk_size, n_warps>(
                /*matrix =*/ scores_chunk,
                /*column =*/ scores_max_local,
                /*ldm =*/ scores_chunk_ldm,
                chunk_size_q_real,
                chunk_size_k_real
            );
        }
        
        __syncthreads();

        if (kv_row_chunk != 0) {
            // denom_local = scores_chunk.sum(dim=1)
            threadblock_row_sum<chunk_size, chunk_size, n_warps, /*copy_to_vec=*/true>(
                /*mat = */ scores_chunk,
                /*vec = */ denom_local,
                /*aux_memory = */ aux_mem,
                /*ldm = */ scores_chunk_ldm,
                /*ldm_aux = */ reduce_sum_ldm
            );
        }
            
        __syncthreads();

        if (kv_row_chunk != 0) {
            // numer_local = scores_chunk @ values_chunk
            threadblock_gemm_k_real< /*m=*/chunk_size, /*n=*/head_dim, /*k=*/chunk_size, n_warps, 
                            /*transpose_b=*/false, /*alpha_is_one=*/true, /*preloaded_a_frags=*/false >(
                /*mat_a_frags = */ nullptr,
                /*mat_a = */ scores_chunk,
                /*mat_b = */ &values[batch_offset_k + kv_row * num_features + head_col],
                /*mat_c = */ numer_local,
                /*m_real = */ chunk_size_q_real,
                /*n_real = */ head_dim,
                /*k_real = */ chunk_size_k_real,
                /*lda = */ scores_chunk_ldm,
                /*ldb = */ num_features,
                /*ldc = */ numer_local_ldm
            );
        }
        
        __syncthreads();
        
        if (kv_row_chunk != 0) {
            threadblock_aggregate_softmax<chunk_size, head_dim, n_warps>(
                /*numer_a = */ numer,
                /*denom_a = */ denom,
                /*max_a = */ scores_max,
                /*numer_b = */ numer_local,
                /*denom_b = */ denom_local,
                /*max_b = */ scores_max_local,
                /*lda =*/ numer_ldm,
                /*ldb =*/ numer_local_ldm,
                /*aux =*/ aux_mem,
                chunk_size_q_real
            );
        }
    }

    __syncthreads();
    
    // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {

    //     printf("numer: \n");
    //     for (int i = 0; i < chunk_size; i++) {
    //         for (int j = 0; j < head_dim; j++) {
    //             printf("%.1f ", __half2float(numer[i * queries_chunk_ldm + j]));
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {

    //     printf("denom: \n");
    //     for (int i = 0; i < chunk_size; i++) {
    //         printf("%.1f ", __half2float(denom[i]));
    //     }
    //     printf("\n");
    // }

    threadblock_divide_and_store_chunk<chunk_size, head_dim, n_warps>(
        /*numer = */ numer,
        /*denom = */ denom,
        /*dst = */ &output[batch_offset_q + q_row * num_features + head_col],
        /*lds = */ numer_ldm,
        /*ldd = */ num_features,
        /*seq_len*/chunk_size_q_real
    );
}

template <
    uint32_t head_dim,
    uint32_t chunk_size_q,
    uint32_t chunk_size_k,
    uint32_t n_warps
>
__global__
__launch_bounds__(256) void attention_kernel_256(
    uint32_t batch_size,
    uint32_t seq_len_q,
    uint32_t seq_len_k,
    uint32_t num_features,
    const half_t* __restrict__ queries,
    const half_t* __restrict__ keys,
    const half_t* __restrict__ values,
    const half_t* __restrict__ mask,
    half_t* __restrict__ output
) {
    
    constexpr uint32_t mat_skew = 8;
    // constexpr uint32_t common_ldm = head_dim + mat_skew;
    constexpr uint32_t reduce_max_ldm = 2 * warp_size * n_warps / chunk_size_k;

    constexpr uint32_t chunk_frags = chunk_size_q / 16;
    constexpr uint32_t head_frags = head_dim / 16;

    constexpr uint32_t max_size = chunk_size_k < head_dim ? head_dim : chunk_size_k;

    // // static_assert(chunk_size <= head_dim);
    // static_assert(chunk_size <= warp_size * n_warps);                     // Column operations should be done in one iter (or less)
    // static_assert(chunk_size * head_dim % (warp_size * n_warps) == 0);    // Matrix operations should be done in several iters
    // static_assert(chunk_size * chunk_size % (warp_size * n_warps) == 0);
    // static_assert(reduce_max_ldm <= 16);

    half_t* queries_chunk;      // matrix (chunk_size, head_dim)
    half_t* scores_chunk;       // matrix (chunk_size, chunk_size)
    half_t* numer_local;        // matrix (chunk_size, head_dim)
    half_t* numer;              // matrix (chunk_size, head_dim)
    half_t* aux_mem;            // matrix (chunk_size, 16)

    constexpr uint32_t numer_ldm = head_dim + mat_skew;
    constexpr uint32_t queries_chunk_ldm = numer_ldm;
    constexpr uint32_t numer_local_ldm = max_size + mat_skew;
    constexpr uint32_t scores_chunk_ldm = numer_local_ldm;
    constexpr uint32_t reduce_sum_ldm = chunk_size_k + mat_skew;

    half_t* scores_max_local;   // vector (chunk_size,)
    half_t* denom_local;        // vector (chunk_size,)
    half_t* denom;              // vector (chunk_size,)
    half_t* scores_max;         // vector (chunk_size,)

    extern __shared__ half_t shmem[];

    distribute_shared_mem<half_t,
        chunk_size_q * numer_ldm,                      // numer / queries_chunk
        (chunk_size_q + 16) * numer_local_ldm,         // numer_local / scores_chunk
        16 * reduce_sum_ldm,                         // aux_mem
        chunk_size_k,                                  // scores_max_local
        chunk_size_k,                                  // denom
        chunk_size_k,                                  // denom_local
        chunk_size_k                                   // scores_max
    >(
        shmem,
        &numer,
        &numer_local,
        &aux_mem,
        &scores_max_local,
        &denom,
        &denom_local,
        &scores_max
    );

    queries_chunk = numer;
    scores_chunk = &numer_local[16 * numer_local_ldm];
    // const uint32_t chunk_size = chunk_size_k;
    // const uint32_t chunk_size_q = 16;
    // const uint32_t chunk_size_k = chunk_size;
    const uint32_t num_chanks_k = (seq_len_k + chunk_size_k - 1) / chunk_size_k;
    const uint32_t chunk_size_q_remain = seq_len_q % chunk_size_q;
    const uint32_t chunk_size_k_remain = seq_len_k % chunk_size_k;

    const uint32_t batch_idx = blockIdx.x;
    const uint32_t q_row_chunk = blockIdx.y;
    const uint32_t head_col_chunk = blockIdx.z;
    // seq_len of  q, k and v may not be the same.
    const uint32_t batch_offset_q = batch_idx * seq_len_q * num_features;
    const uint32_t batch_offset_k = batch_idx * seq_len_k * num_features;
    const uint32_t batch_offset_mask = batch_idx * seq_len_q * seq_len_k;

    // row / col in the matrix of fixed batch_idx (seq_len, num_features)
    const uint32_t q_row = q_row_chunk * chunk_size_q;
    const uint32_t head_col = head_col_chunk * head_dim;
    const uint32_t chunk_size_q_real = (q_row_chunk == gridDim.y - 1 && chunk_size_q_remain > 0) ? chunk_size_q_remain : chunk_size_q;

    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, rocwmma_half, rocwmma::row_major> query_frags[chunk_frags][head_frags];
    // batch_size, sequence_len, num_heads, head_dim
    // Load query chunk into shared memory
    threadblock_load_q_1xhead_dim<chunk_size_q, head_dim, n_warps>(
        /*src = */ &queries[batch_offset_q + q_row * num_features + head_col],
        /*dst = */ queries_chunk,
        /*lds = */ num_features,
        /*ldd = */ queries_chunk_ldm,
        /*seq_len_q*/chunk_size_q_real
    );

    // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     printf("queries_chunk: \n");
    //     for (int i = 0; i < chunk_size_q; i++) {
    //         for (int j = 0; j < head_dim; j++) {
    //             printf("%.1f ", __half2float(queries_chunk[i * queries_chunk_ldm + j]));
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    
    __syncthreads();

    // Move query chunk into warps' fragments, query_chunk is freed here and can be used later as numer
    #pragma unroll
    for (uint32_t row_frag = 0; row_frag < chunk_frags; ++row_frag) {
        #pragma unroll
        for (uint32_t col_frag = 0; col_frag < head_frags; ++col_frag) {
            rocwmma::load_matrix_sync(query_frags[row_frag][col_frag], &queries_chunk[16 * (row_frag * queries_chunk_ldm + col_frag)], queries_chunk_ldm);
        }
    }

    #pragma unroll(1)
    for (uint32_t kv_row_chunk = 0; kv_row_chunk < num_chanks_k; kv_row_chunk++) {
        const uint32_t kv_row = kv_row_chunk * chunk_size_k;
        uint32_t chunk_size_k_real = chunk_size_k;
        if (kv_row_chunk == num_chanks_k - 1 && chunk_size_k_remain > 0) {
            chunk_size_k_real = chunk_size_k_remain;
        }
        // scores_chunk = queries_chunk @ keys_chunk.T / sqrt(head_dim)
        threadblock_gemm_k_real< /*m=*/chunk_size_q, /*n=*/chunk_size_k, /*k=*/head_dim, n_warps,
                        /*transpose_b=*/true, /*alpha_is_one=*/false, /*preloaded_a_frags=*/true >(
            /*mat_a_frags=*/ query_frags,
            /*mat_a =*/ nullptr,
            /*mat_b =*/ &keys[batch_offset_k + kv_row * num_features + head_col],
            /*mat_c =*/ scores_chunk,
                      chunk_size_q_real,
                      chunk_size_k_real,
                      head_dim,
            /*lda =*/ 0,
            /*ldb =*/ num_features,
            /*ldc =*/ scores_chunk_ldm,
            /*alpha =*/ __float2half(rsqrtf(head_dim))
        );

        // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("scores_chunk: \n");
        //     for (int i = 0; i < chunk_size_q; i++) {
        //         for (int j = 0; j < chunk_size_k; j++) {
        //             printf("%.1f ", __half2float(scores_chunk[i * scores_chunk_ldm + j]));
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }

        __syncthreads();


        if (kv_row_chunk == 0) {
            // First iter: write directly to numer, denom, scores_max, no aggregation

            if (mask != nullptr) {
                threadblock_ewise_sum<chunk_size_q, chunk_size_k, n_warps>(
                    /*mat_a =*/ scores_chunk,
                    /*mat_b =*/ &mask[kv_row],
                    /*lda =*/ scores_chunk_ldm,
                    seq_len_k,
                    chunk_size_q_real,
                    chunk_size_k_real
                );
            }
        }
        if (kv_row_chunk == 0) {
            // scores_max = max(scores_chunk, dim=1)
            threadblock_row_max<chunk_size_q, chunk_size_k, n_warps>(
                /*matrix =*/ scores_chunk,
                /*column =*/ scores_max,
                /*aux_memory =*/ aux_mem,
                /*ldm =*/ scores_chunk_ldm,
                /*ldm_aux =*/ reduce_max_ldm,
                chunk_size_q_real,
                chunk_size_k_real
            );
        }

        __syncthreads();

        // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("scores_max: \n");
        //     for (int i = 0; i < chunk_size_k; i++) {
        //         printf("%.1f ", __half2float(scores_max[i]));
        //     }
        //     printf("\n");
        // }
        
        if (kv_row_chunk == 0) {
            // scores_chunk = exp(scores_chunk - scores_max[:, None])
            threadblock_row_broadcast_diff_and_exp<chunk_size_q, chunk_size_k, n_warps>(
                /*matrix =*/ scores_chunk,
                /*column =*/ scores_max,
                /*ldm =*/ scores_chunk_ldm,
                chunk_size_q_real,
                chunk_size_k_real
            );
        }

        // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("diff_and_exp: \n");
        //     for (int i = 0; i < chunk_size_q; i++) {
        //         for (int j = 0; j < chunk_size_k; j++) {
        //             printf("%.1f ", __half2float(scores_chunk[i * scores_chunk_ldm + j]));
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }

        __syncthreads();
        
        if (kv_row_chunk == 0) {
            // denom = scores_chunk.sum(dim=1)
            threadblock_row_sum<chunk_size_q, chunk_size_k, n_warps>(
                /*mat = */ scores_chunk,
                /*vec = */ denom,
                /*aux_memory = */ aux_mem,
                /*ldm = */ scores_chunk_ldm,
                /*ldm_aux = */ reduce_sum_ldm,
                chunk_size_q_real,
                chunk_size_k_real
            );
        }

        // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("denom: \n");
        //     for (int i = 0; i < chunk_size_k; i++) {
        //         printf("%.1f ", __half2float(denom[i]));
        //     }
        //     printf("\n");

        //     printf("aux_mem: \n");
        //     for (int i = 0; i < chunk_size_k; i++) {
        //         printf("%.1f ", __half2float(aux_mem[i]));
        //     }
        //     printf("\n");
        // }
        
        __syncthreads();
        
        if (kv_row_chunk == 0) {
            // numer = scores_chunk @ values_chunk
            threadblock_gemm_k_real< /*m=*/chunk_size_q, /*n=*/head_dim, /*k=*/chunk_size_k, n_warps, 
                            /*transpose_b=*/false, /*alpha_is_one=*/true, /*preloaded_a_frags=*/false >(
                /*mat_a_frags = */ nullptr,
                /*mat_a = */ scores_chunk,
                /*mat_b = */ &values[batch_offset_k + head_col],
                /*mat_c = */ numer,
                /*m_real = */ chunk_size_q_real,
                /*n_real = */ head_dim,
                /*k_real = */ chunk_size_k_real,
                /*lda = */ scores_chunk_ldm,
                /*ldb = */ num_features,
                /*ldc = */ numer_ldm
            );

        }

        __syncthreads();

        // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     // printf("scores_chunk: \n");
        //     // for (int i = 0; i < chunk_size_q; i++) {
        //     //     for (int j = 0; j < chunk_size_k_real; j++) {
        //     //         printf("%.1f ", __half2float(scores_chunk[i * scores_chunk_ldm + j]));
        //     //     }
        //     //     printf("\n");
        //     // }
        //     // printf("\n");
        //     // printf("values: \n");
        //     // for (int i = 0; i < chunk_size_k_real; i++) {
        //     //     for (int j = 0; j < head_dim; j++) {
        //     //         printf("%.1f ", __half2float((&values[batch_offset_k + head_col])[i * num_features + j]));
        //     //     }
        //     //     printf("\n");
        //     // }
        //     // printf("\n");
        //     // printf("numer: \n");
        //     // for (int i = 0; i < chunk_size_q; i++) {
        //     //     for (int j = 0; j < head_dim; j++) {
        //     //         printf("%.1f ", __half2float(numer[i * numer_ldm + j]));
        //     //     }
        //     //     printf("\n");
        //     // }
        //     // printf("\n");
        // }

        // Second and further iterations, write to temprorary numer_local, denom_local, scores_max_local, then aggregate with prev values

        // scores_max = max(scores_chunk, dim=1)
        if (kv_row_chunk != 0) {
            threadblock_row_max<chunk_size_q, chunk_size_k, n_warps>(
                /*matrix =*/ scores_chunk,
                /*column =*/ scores_max_local,
                /*aux_memory =*/ aux_mem,
                /*ldm =*/ scores_chunk_ldm,
                /*ldm_aux =*/ reduce_max_ldm,
                chunk_size_q_real,
                chunk_size_k_real
            );
        }

        __syncthreads();

        if (kv_row_chunk != 0) {
            // scores_chunk = exp(scores_chunk - scores_max[:, None])
            threadblock_row_broadcast_diff_and_exp<chunk_size_q, chunk_size_k, n_warps>(
                /*matrix =*/ scores_chunk,
                /*column =*/ scores_max_local,
                /*ldm =*/ scores_chunk_ldm,
                chunk_size_q_real,
                chunk_size_k_real
            );
        }
        
        __syncthreads();

        if (kv_row_chunk != 0) {
            // denom_local = scores_chunk.sum(dim=1)
            threadblock_row_sum<chunk_size_q, chunk_size_k, n_warps, /*copy_to_vec=*/true>(
                /*mat = */ scores_chunk,
                /*vec = */ denom_local,
                /*aux_memory = */ aux_mem,
                /*ldm = */ scores_chunk_ldm,
                /*ldm_aux = */ reduce_sum_ldm,
                chunk_size_q_real,
                chunk_size_k_real
            );
        }
            
        __syncthreads();

        if (kv_row_chunk != 0) {
            // numer_local = scores_chunk @ values_chunk
            threadblock_gemm_k_real< /*m=*/chunk_size_q, /*n=*/head_dim, /*k=*/chunk_size_k, n_warps, 
                            /*transpose_b=*/false, /*alpha_is_one=*/true, /*preloaded_a_frags=*/false >(
                /*mat_a_frags = */ nullptr,
                /*mat_a = */ scores_chunk,
                /*mat_b = */ &values[batch_offset_k + kv_row * num_features + head_col],
                /*mat_c = */ numer_local,
                /*m_real = */ chunk_size_q_real,
                /*n_real = */ head_dim,
                /*k_real = */ chunk_size_k_real,
                /*lda = */ scores_chunk_ldm,
                /*ldb = */ num_features,
                /*ldc = */ numer_local_ldm
            );
        }
        
        __syncthreads();
        
        if (kv_row_chunk != 0) {
            threadblock_aggregate_softmax<chunk_size_k, head_dim, n_warps>(
                /*numer_a = */ numer,
                /*denom_a = */ denom,
                /*max_a = */ scores_max,
                /*numer_b = */ numer_local,
                /*denom_b = */ denom_local,
                /*max_b = */ scores_max_local,
                /*lda =*/ numer_ldm,
                /*ldb =*/ numer_local_ldm,
                /*aux =*/ aux_mem,
                chunk_size_q_real
            );
        }
    }

    __syncthreads();
    
    // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {

    //     printf("numer: \n");
    //     for (int i = 0; i < chunk_size_q; i++) {
    //         for (int j = 0; j < head_dim; j++) {
    //             printf("%.1f ", __half2float(numer[i * numer_ldm + j]));
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {

    //     printf("denom: \n");
    //     for (int i = 0; i < chunk_size_q; i++) {
    //         printf("%.1f ", __half2float(denom[i]));
    //     }
    //     printf("\n");
    // }

    threadblock_divide_and_store_chunk<chunk_size_q, head_dim, n_warps>(
        /*numer = */ numer,
        /*denom = */ denom,
        /*dst = */ &output[batch_offset_q + q_row * num_features + head_col],
        /*lds = */ numer_ldm,
        /*ldd = */ num_features,
        /*seq_len*/chunk_size_q_real
    );
}

template <uint32_t head_dim, uint32_t chunk_size>
void launch_attention_kernel(
    uint32_t batch_size,
    uint32_t seq_len_q,
    uint32_t seq_len_k,
    uint32_t num_features,
    const half_t* queries,
    const half_t* keys,
    const half_t* values,
    const half_t* mask,
    half_t* output,
    bool syncronize = false
) {
    if (seq_len_k < 24) {
        constexpr uint32_t n_warps = chunk_size < head_dim ? chunk_size / 16 : head_dim / 16;
        constexpr uint32_t max_size = chunk_size > head_dim ? chunk_size : head_dim;
        constexpr uint32_t shared_mem_size = ( chunk_size * (head_dim + 8) + (chunk_size + 16) * (max_size + 8) + 
                                            16 * (chunk_size + 8) + 4 * chunk_size ) * sizeof(half_t);
        if (warp_size * n_warps >= 1024) {
            printf("kernel thread in block is greater than 1024!\n");
            return;
        }
        // Call attention kernel
        dim3 threads(warp_size, n_warps, 1);
        dim3 blocks(batch_size, (seq_len_q + chunk_size - 1) / chunk_size, num_features / head_dim);
        
        attention_kernel_64<head_dim, chunk_size, n_warps><<<dim3(blocks), dim3(threads), shared_mem_size>>>(
            batch_size, seq_len_q, seq_len_k, num_features, 
            queries, keys, values, mask, output
        );
    } else {
        constexpr uint32_t n_warps = 4;
        constexpr uint32_t max_size = chunk_size > head_dim ? chunk_size : head_dim;
        constexpr uint32_t seq_len_k_tiling_size = 128;
        constexpr uint32_t chunk_size_k = 128;
        
        // constexpr uint32_t numer_ldm = head_dim + mat_skew;
        // constexpr uint32_t queries_chunk_ldm = numer_ldm;
        // constexpr uint32_t numer_local_ldm = chunk_size_k + mat_skew;
        // constexpr uint32_t scores_chunk_ldm = numer_local_ldm;
        // constexpr uint32_t reduce_sum_ldm = chunk_size_k + mat_skew;
        //  chunk_size_q * numer_ldm,                      // numer / queries_chunk
        // (chunk_size_q + 16) * numer_local_ldm,         // numer_local / scores_chunk
        // 16 * reduce_sum_ldm,                         // aux_mem
        // chunk_size_k,                                  // scores_max_local
        // chunk_size_k,                                  // denom
        // chunk_size_k,                                  // denom_local
        // chunk_size_k                                   // scores_max

        constexpr uint32_t queries_chunk_size = chunk_size * (head_dim + 8);
        constexpr uint32_t scores_chunk_size = (chunk_size + 16) * (seq_len_k_tiling_size + 8);
        constexpr uint32_t aux_size = 16 * (seq_len_k_tiling_size + 8);
        constexpr uint32_t scores_max_local_size = seq_len_k_tiling_size;
        constexpr uint32_t denom_size = seq_len_k_tiling_size;
        constexpr uint32_t denom_local_size = seq_len_k_tiling_size;
        constexpr uint32_t scores_max_size = seq_len_k_tiling_size;


        constexpr uint32_t shared_mem_size = (queries_chunk_size + scores_chunk_size + aux_size + 
                                            scores_max_local_size + denom_size + denom_local_size + scores_max_size) * sizeof(half_t);
        if (warp_size * n_warps >= 1024) {
            printf("kernel thread in block is greater than 1024!\n");
            return;
        }
        // Call attention kernel
        dim3 threads(warp_size, n_warps, 1);
        dim3 blocks(batch_size, (seq_len_q + chunk_size - 1) / chunk_size, num_features / head_dim);
        
        attention_kernel_256<head_dim, chunk_size, chunk_size_k, n_warps><<<dim3(blocks), dim3(threads), shared_mem_size>>>(
            batch_size, seq_len_q, seq_len_k, num_features, 
            queries, keys, values, mask, output
        );
    }

    hipDeviceSynchronize();

    if (syncronize) {
        CHECK_CUDA_ERROR(hipDeviceSynchronize());
        CHECK_LAST_CUDA_ERROR();        
    }
}


template <
    uint32_t head_dim,
    uint32_t chunk_size,
    uint32_t n_warps
>
__global__
__launch_bounds__(64) void attention_trans_kernel_64(
    uint32_t batch_size,
    uint32_t seq_len_q,
    uint32_t seq_len_k,
    uint32_t num_features,
    const half_t* __restrict__ queries,
    const half_t* __restrict__ keys,
    const half_t* __restrict__ values,
    const half_t* __restrict__ mask,
    half_t* __restrict__ output
) {

    constexpr uint32_t mat_skew = 8;
    // constexpr uint32_t common_ldm = head_dim + mat_skew;
    constexpr uint32_t reduce_max_ldm = 2 * warp_size * n_warps / chunk_size;

    constexpr uint32_t chunk_frags = chunk_size / 16;
    constexpr uint32_t head_frags = head_dim / 16;

    constexpr uint32_t max_size = chunk_size < head_dim ? head_dim : chunk_size;

    // static_assert(chunk_size <= head_dim);
    static_assert(chunk_size <= warp_size * n_warps);                     // Column operations should be done in one iter (or less)
    static_assert(chunk_size * head_dim % (warp_size * n_warps) == 0);    // Matrix operations should be done in several iters
    static_assert(chunk_size * chunk_size % (warp_size * n_warps) == 0);
    static_assert(reduce_max_ldm <= 16);

    half_t* queries_chunk;      // matrix (chunk_size, head_dim)
    half_t* scores_chunk;       // matrix (chunk_size, chunk_size)
    half_t* numer_local;        // matrix (chunk_size, head_dim)
    half_t* numer;              // matrix (chunk_size, head_dim)
    half_t* aux_mem;            // matrix (chunk_size, 16)

    constexpr uint32_t numer_ldm = head_dim + mat_skew;
    constexpr uint32_t queries_chunk_ldm = numer_ldm;
    constexpr uint32_t numer_local_ldm = max_size + mat_skew;
    constexpr uint32_t scores_chunk_ldm = numer_local_ldm;
    constexpr uint32_t reduce_sum_ldm = chunk_size + mat_skew;

    half_t* scores_max_local;   // vector (chunk_size,)
    half_t* denom_local;        // vector (chunk_size,)
    half_t* denom;              // vector (chunk_size,)
    half_t* scores_max;         // vector (chunk_size,)

    extern __shared__ half_t shmem[];

    distribute_shared_mem<half_t,
        chunk_size * numer_ldm,                      // numer / queries_chunk
        (chunk_size + 16) * numer_local_ldm,         // numer_local / scores_chunk
        16 * reduce_sum_ldm,                         // aux_mem
        chunk_size,                                  // scores_max_local
        chunk_size,                                  // denom
        chunk_size,                                  // denom_local
        chunk_size                                   // scores_max
    >(
        shmem,
        &numer,
        &numer_local,
        &aux_mem,
        &scores_max_local,
        &denom,
        &denom_local,
        &scores_max
    );

    queries_chunk = numer;
    scores_chunk = &numer_local[16 * numer_local_ldm];
    
    // const uint32_t chunk_size_q = 16;
    // const uint32_t chunk_size_k = chunk_size;
    const uint32_t num_chanks_k = (seq_len_k + chunk_size - 1) / chunk_size;
    const uint32_t chunk_size_q_remain = seq_len_q % chunk_size;
    const uint32_t chunk_size_k_remain = seq_len_k % chunk_size;
    
    const uint32_t num_heads = num_features / head_dim;
    const uint32_t batch_idx = blockIdx.x;
    const uint32_t q_row_chunk = blockIdx.y;
    const uint32_t head_col_chunk = batch_idx % num_heads;

    // row / col in the matrix of fixed batch_idx (seq_len, num_features)
    const uint32_t q_row = q_row_chunk * chunk_size;
    const uint32_t head_col = head_col_chunk * head_dim;
    const uint32_t chunk_size_q_real = (q_row_chunk == gridDim.y - 1 && chunk_size_q_remain > 0) ? chunk_size_q_remain : chunk_size;

    const uint32_t batch_offset_q = batch_idx * seq_len_q * head_dim;
    const uint32_t batch_offset_k = batch_idx * seq_len_k * head_dim;
    const uint32_t batch_offset_q_no_trans = batch_idx / num_heads * seq_len_q * num_features;
    const uint32_t batch_offset_mask = 0;

    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, rocwmma_half, rocwmma::row_major> query_frags[chunk_frags][head_frags];
    // batch_size, sequence_len, num_heads, head_dim
    // Load query chunk into shared memory
    threadblock_load_q_1xhead_dim<chunk_size, head_dim, n_warps>(
        /*src = */ &queries[batch_offset_q],
        /*dst = */ queries_chunk,
        /*lds = */ head_dim,
        /*ldd = */ queries_chunk_ldm,
        /*seq_len_q*/chunk_size_q_real
    );

    // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     printf("queries_chunk: \n");
    //     for (int i = 0; i < chunk_size; i++) {
    //         for (int j = 0; j < head_dim; j++) {
    //             printf("%.1f ", __half2float(queries_chunk[i * queries_chunk_ldm + j]));
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    __syncthreads();

    // Move query chunk into warps' fragments, query_chunk is freed here and can be used later as numer
    #pragma unroll
    for (uint32_t row_frag = 0; row_frag < chunk_frags; ++row_frag) {
        #pragma unroll
        for (uint32_t col_frag = 0; col_frag < head_frags; ++col_frag) {
            rocwmma::load_matrix_sync(query_frags[row_frag][col_frag], &queries_chunk[16 * (row_frag * queries_chunk_ldm + col_frag)], queries_chunk_ldm);
        }
    }

    #pragma unroll(1)
    for (uint32_t kv_row_chunk = 0; kv_row_chunk < num_chanks_k; kv_row_chunk++) {
        const uint32_t kv_row = kv_row_chunk * chunk_size;
        uint32_t chunk_size_k_real = chunk_size;
        if (kv_row_chunk == num_chanks_k - 1 && chunk_size_k_remain > 0) {
            chunk_size_k_real = chunk_size_k_remain;
        }
        // scores_chunk = queries_chunk @ keys_chunk.T / sqrt(head_dim)
        threadblock_gemm_k_real< /*m=*/chunk_size, /*n=*/chunk_size, /*k=*/head_dim, n_warps,
                        /*transpose_b=*/true, /*alpha_is_one=*/false, /*preloaded_a_frags=*/true >(
            /*mat_a_frags=*/ query_frags,
            /*mat_a =*/ nullptr,
            /*mat_b =*/ &keys[batch_offset_k + kv_row * head_dim],
            /*mat_c =*/ scores_chunk,
                      chunk_size_q_real,
                      chunk_size_k_real,
                      head_dim,
            /*lda =*/ 0,
            /*ldb =*/ head_dim,
            /*ldc =*/ scores_chunk_ldm,
            /*alpha =*/ __float2half(rsqrtf(head_dim))
        );
        // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("scores_chunk: \n");
        //     for (int i = 0; i < chunk_size; i++) {
        //         for (int j = 0; j < chunk_size; j++) {
        //             printf("%.1f ", __half2float(scores_chunk[i * queries_chunk_ldm + j]));
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }

        __syncthreads();


        if (kv_row_chunk == 0) {
            // First iter: write directly to numer, denom, scores_max, no aggregation

            if (mask != nullptr) {
                threadblock_ewise_sum<chunk_size, chunk_size, n_warps>(
                    /*mat_a =*/ scores_chunk,
                    /*mat_b =*/ &mask[kv_row],
                    /*lda =*/ scores_chunk_ldm,
                    seq_len_k,
                    chunk_size_q_real,
                    chunk_size_k_real
                );
            }
        }
        if (kv_row_chunk == 0) {
            // scores_max = max(scores_chunk, dim=1)
            threadblock_row_max<chunk_size, chunk_size, n_warps>(
                /*matrix =*/ scores_chunk,
                /*column =*/ scores_max,
                /*aux_memory =*/ aux_mem,
                /*ldm =*/ scores_chunk_ldm,
                /*ldm_aux =*/ reduce_max_ldm,
                chunk_size_q_real,
                chunk_size_k_real
            );
        }
        // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("scores_max: \n");
        //     for (int i = 0; i < chunk_size; i++) {
        //         printf("%.1f ", __half2float(scores_max[i]));
        //     }
        //     printf("\n");
        // }

        __syncthreads();
        
        if (kv_row_chunk == 0) {
            // scores_chunk = exp(scores_chunk - scores_max[:, None])
            threadblock_row_broadcast_diff_and_exp<chunk_size, chunk_size, n_warps>(
                /*matrix =*/ scores_chunk,
                /*column =*/ scores_max,
                /*ldm =*/ scores_chunk_ldm,
                chunk_size_q_real,
                chunk_size_k_real
            );
        }
        // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("diff_and_exp: \n");
        //     for (int i = 0; i < chunk_size; i++) {
        //         for (int j = 0; j < chunk_size; j++) {
        //             printf("%.1f ", __half2float(scores_chunk[i * queries_chunk_ldm + j]));
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }

        __syncthreads();
        
        if (kv_row_chunk == 0) {
            // denom = scores_chunk.sum(dim=1)
            threadblock_row_sum<chunk_size, chunk_size, n_warps>(
                /*mat = */ scores_chunk,
                /*vec = */ denom,
                /*aux_memory = */ aux_mem,
                /*ldm = */ scores_chunk_ldm,
                /*ldm_aux = */ reduce_sum_ldm
            );
        }
        
        __syncthreads();
        
        if (kv_row_chunk == 0) {
            // numer = scores_chunk @ values_chunk
            threadblock_gemm_k_real< /*m=*/chunk_size, /*n=*/head_dim, /*k=*/chunk_size, n_warps, 
                            /*transpose_b=*/false, /*alpha_is_one=*/true, /*preloaded_a_frags=*/false >(
                /*mat_a_frags = */ nullptr,
                /*mat_a = */ scores_chunk,
                /*mat_b = */ &values[batch_offset_k + kv_row * head_dim],
                /*mat_c = */ numer,
                /*m_real = */ chunk_size_q_real,
                /*n_real = */ head_dim,
                /*k_real = */ chunk_size_k_real,
                /*lda = */ scores_chunk_ldm,
                /*ldb = */ head_dim,
                /*ldc = */ numer_ldm
            );

        }
        // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {

        //     printf("numer: \n");
        //     for (int i = 0; i < chunk_size; i++) {
        //         for (int j = 0; j < head_dim; j++) {
        //             printf("%.1f ", __half2float(numer[i * queries_chunk_ldm + j]));
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }

        // Second and further iterations, write to temprorary numer_local, denom_local, scores_max_local, then aggregate with prev values

        // scores_max = max(scores_chunk, dim=1)
        if (kv_row_chunk != 0) {
            threadblock_row_max<chunk_size, chunk_size, n_warps>(
                /*matrix =*/ scores_chunk,
                /*column =*/ scores_max_local,
                /*aux_memory =*/ aux_mem,
                /*ldm =*/ scores_chunk_ldm,
                /*ldm_aux =*/ reduce_max_ldm,
                chunk_size_q_real,
                chunk_size_k_real
            );
        }

        __syncthreads();

        if (kv_row_chunk != 0) {
            // scores_chunk = exp(scores_chunk - scores_max[:, None])
            threadblock_row_broadcast_diff_and_exp<chunk_size, chunk_size, n_warps>(
                /*matrix =*/ scores_chunk,
                /*column =*/ scores_max_local,
                /*ldm =*/ scores_chunk_ldm,
                chunk_size_q_real,
                chunk_size_k_real
            );
        }
        
        __syncthreads();

        if (kv_row_chunk != 0) {
            // denom_local = scores_chunk.sum(dim=1)
            threadblock_row_sum<chunk_size, chunk_size, n_warps, /*copy_to_vec=*/true>(
                /*mat = */ scores_chunk,
                /*vec = */ denom_local,
                /*aux_memory = */ aux_mem,
                /*ldm = */ scores_chunk_ldm,
                /*ldm_aux = */ reduce_sum_ldm
            );
        }
            
        __syncthreads();

        if (kv_row_chunk != 0) {
            // numer_local = scores_chunk @ values_chunk
            threadblock_gemm_k_real< /*m=*/chunk_size, /*n=*/head_dim, /*k=*/chunk_size, n_warps, 
                            /*transpose_b=*/false, /*alpha_is_one=*/true, /*preloaded_a_frags=*/false >(
                /*mat_a_frags = */ nullptr,
                /*mat_a = */ scores_chunk,
                /*mat_b = */ &values[batch_offset_k + kv_row * head_dim],
                /*mat_c = */ numer_local,
                /*m_real = */ chunk_size_q_real,
                /*n_real = */ head_dim,
                /*k_real = */ chunk_size_k_real,
                /*lda = */ scores_chunk_ldm,
                /*ldb = */ head_dim,
                /*ldc = */ numer_local_ldm
            );
        }
        
        __syncthreads();
        
        if (kv_row_chunk != 0) {
            threadblock_aggregate_softmax<chunk_size, head_dim, n_warps>(
                /*numer_a = */ numer,
                /*denom_a = */ denom,
                /*max_a = */ scores_max,
                /*numer_b = */ numer_local,
                /*denom_b = */ denom_local,
                /*max_b = */ scores_max_local,
                /*lda =*/ numer_ldm,
                /*ldb =*/ numer_local_ldm,
                /*aux =*/ aux_mem,
                chunk_size_k_real
            );
        }
    }

    __syncthreads();
    
    // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {

    //     printf("numer: \n");
    //     for (int i = 0; i < chunk_size; i++) {
    //         for (int j = 0; j < head_dim; j++) {
    //             printf("%.1f ", __half2float(numer[i * queries_chunk_ldm + j]));
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {

    //     printf("denom: \n");
    //     for (int i = 0; i < chunk_size; i++) {
    //         printf("%.1f ", __half2float(denom[i]));
    //     }
    //     printf("\n");
    // }

    // threadblock_divide_and_store_chunk<chunk_size, head_dim, n_warps>(
    //     /*numer = */ numer,
    //     /*denom = */ denom,
    //     /*dst = */ &output[batch_offset_q],
    //     /*lds = */ numer_ldm,
    //     /*ldd = */ head_dim,
    //     /*seq_len*/chunk_size_q_real
    // );

    threadblock_divide_and_store_chunk<chunk_size, head_dim, n_warps>(
        /*numer = */ numer,
        /*denom = */ denom,
        /*dst = */ &output[batch_offset_q_no_trans + head_col],
        /*lds = */ numer_ldm,
        /*ldd = */ num_features,
        /*seq_len*/chunk_size_q_real
    );
}


template <
    uint32_t head_dim,
    uint32_t chunk_size_q,
    uint32_t chunk_size_k,
    uint32_t n_warps
>
__global__
__launch_bounds__(256) void attention_trans_kernel_256(
    uint32_t batch_size,
    uint32_t seq_len_q,
    uint32_t seq_len_k,
    uint32_t num_features,
    const half_t* __restrict__ queries,
    const half_t* __restrict__ keys,
    const half_t* __restrict__ values,
    const half_t* __restrict__ mask,
    half_t* __restrict__ output
) {
    
    constexpr uint32_t mat_skew = 8;
    // constexpr uint32_t common_ldm = head_dim + mat_skew;
    constexpr uint32_t reduce_max_ldm = 2 * warp_size * n_warps / chunk_size_k;

    constexpr uint32_t chunk_frags = chunk_size_q / 16;
    constexpr uint32_t head_frags = head_dim / 16;

    constexpr uint32_t max_size = chunk_size_k < head_dim ? head_dim : chunk_size_k;

    half_t* queries_chunk;      // matrix (chunk_size, head_dim)
    half_t* scores_chunk;       // matrix (chunk_size, chunk_size)
    half_t* numer_local;        // matrix (chunk_size, head_dim)
    half_t* numer;              // matrix (chunk_size, head_dim)
    half_t* aux_mem;            // matrix (chunk_size, 16)

    constexpr uint32_t numer_ldm = head_dim + mat_skew;
    constexpr uint32_t queries_chunk_ldm = numer_ldm;
    constexpr uint32_t numer_local_ldm = max_size + mat_skew;
    constexpr uint32_t scores_chunk_ldm = numer_local_ldm;
    constexpr uint32_t reduce_sum_ldm = chunk_size_k + mat_skew;

    half_t* scores_max_local;   // vector (chunk_size,)
    half_t* denom_local;        // vector (chunk_size,)
    half_t* denom;              // vector (chunk_size,)
    half_t* scores_max;         // vector (chunk_size,)

    extern __shared__ half_t shmem[];

    distribute_shared_mem<half_t,
        chunk_size_q * numer_ldm,                      // numer / queries_chunk
        (chunk_size_q + 16) * numer_local_ldm,         // numer_local / scores_chunk
        16 * reduce_sum_ldm,                         // aux_mem
        chunk_size_k,                                  // scores_max_local
        chunk_size_k,                                  // denom
        chunk_size_k,                                  // denom_local
        chunk_size_k                                   // scores_max
    >(
        shmem,
        &numer,
        &numer_local,
        &aux_mem,
        &scores_max_local,
        &denom,
        &denom_local,
        &scores_max
    );

    queries_chunk = numer;
    scores_chunk = &numer_local[16 * numer_local_ldm];
    // const uint32_t chunk_size = chunk_size_k;
    // const uint32_t chunk_size_q = 16;
    // const uint32_t chunk_size_k = chunk_size;
    const uint32_t num_chanks_k = (seq_len_k + chunk_size_k - 1) / chunk_size_k;
    const uint32_t chunk_size_q_remain = seq_len_q % chunk_size_q;
    const uint32_t chunk_size_k_remain = seq_len_k % chunk_size_k;

    const uint32_t num_heads = num_features / head_dim;

    const uint32_t batch_idx = blockIdx.x;
    const uint32_t q_row_chunk = blockIdx.y;
    // const uint32_t head_col_chunk = blockIdx.z;
    const uint32_t head_col_chunk = batch_idx % num_heads;
    // seq_len of  q, k and v may not be the same.
    const uint32_t batch_offset_q = batch_idx * seq_len_q * head_dim;
    const uint32_t batch_offset_q_no_trans = batch_idx / num_heads * seq_len_q * num_features;
    const uint32_t batch_offset_k = batch_idx * seq_len_k * head_dim;
    const uint32_t batch_offset_mask = batch_idx * seq_len_q * seq_len_k;

    // row / col in the matrix of fixed batch_idx (seq_len, num_features)
    const uint32_t q_row = q_row_chunk * chunk_size_q;
    const uint32_t head_col = head_col_chunk * head_dim;
    const uint32_t chunk_size_q_real = (q_row_chunk == gridDim.y - 1 && chunk_size_q_remain > 0) ? chunk_size_q_remain : chunk_size_q;

    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, rocwmma_half, rocwmma::row_major> query_frags[chunk_frags][head_frags];
    // batch_size, sequence_len, num_heads, head_dim
    // Load query chunk into shared memory
    threadblock_load_q_1xhead_dim<chunk_size_q, head_dim, n_warps>(
        /*src = */ &queries[batch_offset_q],
        /*dst = */ queries_chunk,
        /*lds = */ head_dim,
        /*ldd = */ queries_chunk_ldm,
        /*seq_len_q*/chunk_size_q_real
    );

    // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     printf("queries_chunk: \n");
    //     for (int i = 0; i < chunk_size_q; i++) {
    //         for (int j = 0; j < head_dim; j++) {
    //             printf("%.1f ", __half2float(queries_chunk[i * queries_chunk_ldm + j]));
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    
    __syncthreads();

    // Move query chunk into warps' fragments, query_chunk is freed here and can be used later as numer
    #pragma unroll
    for (uint32_t row_frag = 0; row_frag < chunk_frags; ++row_frag) {
        #pragma unroll
        for (uint32_t col_frag = 0; col_frag < head_frags; ++col_frag) {
            rocwmma::load_matrix_sync(query_frags[row_frag][col_frag], &queries_chunk[16 * (row_frag * queries_chunk_ldm + col_frag)], queries_chunk_ldm);
        }
    }

    #pragma unroll(1)
    for (uint32_t kv_row_chunk = 0; kv_row_chunk < num_chanks_k; kv_row_chunk++) {
        const uint32_t kv_row = kv_row_chunk * chunk_size_k;
        uint32_t chunk_size_k_real = chunk_size_k;
        if (kv_row_chunk == num_chanks_k - 1 && chunk_size_k_remain > 0) {
            chunk_size_k_real = chunk_size_k_remain;
        }
        // scores_chunk = queries_chunk @ keys_chunk.T / sqrt(head_dim)
        threadblock_gemm_k_real< /*m=*/chunk_size_q, /*n=*/chunk_size_k, /*k=*/head_dim, n_warps,
                        /*transpose_b=*/true, /*alpha_is_one=*/false, /*preloaded_a_frags=*/true >(
            /*mat_a_frags=*/ query_frags,
            /*mat_a =*/ nullptr,
            /*mat_b =*/ &keys[batch_offset_k + kv_row * head_dim],
            /*mat_c =*/ scores_chunk,
                      chunk_size_q_real,
                      chunk_size_k_real,
                      head_dim,
            /*lda =*/ 0,
            /*ldb =*/ head_dim,
            /*ldc =*/ scores_chunk_ldm,
            /*alpha =*/ __float2half(rsqrtf(head_dim))
        );

        // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("scores_chunk: \n");
        //     for (int i = 0; i < chunk_size_q; i++) {
        //         for (int j = 0; j < chunk_size_k; j++) {
        //             printf("%.1f ", __half2float(scores_chunk[i * scores_chunk_ldm + j]));
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }

        __syncthreads();


        if (kv_row_chunk == 0) {
            // First iter: write directly to numer, denom, scores_max, no aggregation

            if (mask != nullptr) {
                threadblock_ewise_sum<chunk_size_q, chunk_size_k, n_warps>(
                    /*mat_a =*/ scores_chunk,
                    /*mat_b =*/ &mask[kv_row],
                    /*lda =*/ scores_chunk_ldm,
                    seq_len_k,
                    chunk_size_q_real,
                    chunk_size_k_real
                );
            }
        }
        if (kv_row_chunk == 0) {
            // scores_max = max(scores_chunk, dim=1)
            threadblock_row_max<chunk_size_q, chunk_size_k, n_warps>(
                /*matrix =*/ scores_chunk,
                /*column =*/ scores_max,
                /*aux_memory =*/ aux_mem,
                /*ldm =*/ scores_chunk_ldm,
                /*ldm_aux =*/ reduce_max_ldm,
                chunk_size_q_real,
                chunk_size_k_real
            );
        }

        __syncthreads();

        // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("scores_max: \n");
        //     for (int i = 0; i < chunk_size_k; i++) {
        //         printf("%.1f ", __half2float(scores_max[i]));
        //     }
        //     printf("\n");
        // }
        
        if (kv_row_chunk == 0) {
            // scores_chunk = exp(scores_chunk - scores_max[:, None])
            threadblock_row_broadcast_diff_and_exp<chunk_size_q, chunk_size_k, n_warps>(
                /*matrix =*/ scores_chunk,
                /*column =*/ scores_max,
                /*ldm =*/ scores_chunk_ldm,
                chunk_size_q_real,
                chunk_size_k_real
            );
        }

        // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("diff_and_exp: \n");
        //     for (int i = 0; i < chunk_size_q; i++) {
        //         for (int j = 0; j < chunk_size_k; j++) {
        //             printf("%.1f ", __half2float(scores_chunk[i * scores_chunk_ldm + j]));
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }

        __syncthreads();
        
        if (kv_row_chunk == 0) {
            // denom = scores_chunk.sum(dim=1)
            threadblock_row_sum<chunk_size_q, chunk_size_k, n_warps>(
                /*mat = */ scores_chunk,
                /*vec = */ denom,
                /*aux_memory = */ aux_mem,
                /*ldm = */ scores_chunk_ldm,
                /*ldm_aux = */ reduce_sum_ldm,
                chunk_size_q_real,
                chunk_size_k_real
            );
        }

        // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("denom: \n");
        //     for (int i = 0; i < chunk_size_k; i++) {
        //         printf("%.1f ", __half2float(denom[i]));
        //     }
        //     printf("\n");

        //     printf("aux_mem: \n");
        //     for (int i = 0; i < chunk_size_k; i++) {
        //         printf("%.1f ", __half2float(aux_mem[i]));
        //     }
        //     printf("\n");
        // }
        
        __syncthreads();
        
        if (kv_row_chunk == 0) {
            // numer = scores_chunk @ values_chunk
            threadblock_gemm_k_real< /*m=*/chunk_size_q, /*n=*/head_dim, /*k=*/chunk_size_k, n_warps, 
                            /*transpose_b=*/false, /*alpha_is_one=*/true, /*preloaded_a_frags=*/false >(
                /*mat_a_frags = */ nullptr,
                /*mat_a = */ scores_chunk,
                /*mat_b = */ &values[batch_offset_k + kv_row * head_dim],
                /*mat_c = */ numer,
                /*m_real = */ chunk_size_q_real,
                /*n_real = */ head_dim,
                /*k_real = */ chunk_size_k_real,
                /*lda = */ scores_chunk_ldm,
                /*ldb = */ head_dim,
                /*ldc = */ numer_ldm
            );

        }

        __syncthreads();

        // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     // printf("scores_chunk: \n");
        //     // for (int i = 0; i < chunk_size_q; i++) {
        //     //     for (int j = 0; j < chunk_size_k_real; j++) {
        //     //         printf("%.1f ", __half2float(scores_chunk[i * scores_chunk_ldm + j]));
        //     //     }
        //     //     printf("\n");
        //     // }
        //     // printf("\n");
        //     // printf("values: \n");
        //     // for (int i = 0; i < chunk_size_k_real; i++) {
        //     //     for (int j = 0; j < head_dim; j++) {
        //     //         printf("%.1f ", __half2float((&values[batch_offset_k + head_col])[i * num_features + j]));
        //     //     }
        //     //     printf("\n");
        //     // }
        //     // printf("\n");
        //     // printf("numer: \n");
        //     // for (int i = 0; i < chunk_size_q; i++) {
        //     //     for (int j = 0; j < head_dim; j++) {
        //     //         printf("%.1f ", __half2float(numer[i * numer_ldm + j]));
        //     //     }
        //     //     printf("\n");
        //     // }
        //     // printf("\n");
        // }

        // Second and further iterations, write to temprorary numer_local, denom_local, scores_max_local, then aggregate with prev values

        // scores_max = max(scores_chunk, dim=1)
        if (kv_row_chunk != 0) {
            threadblock_row_max<chunk_size_q, chunk_size_k, n_warps>(
                /*matrix =*/ scores_chunk,
                /*column =*/ scores_max_local,
                /*aux_memory =*/ aux_mem,
                /*ldm =*/ scores_chunk_ldm,
                /*ldm_aux =*/ reduce_max_ldm,
                chunk_size_q_real,
                chunk_size_k_real
            );
        }

        __syncthreads();

        if (kv_row_chunk != 0) {
            // scores_chunk = exp(scores_chunk - scores_max[:, None])
            threadblock_row_broadcast_diff_and_exp<chunk_size_q, chunk_size_k, n_warps>(
                /*matrix =*/ scores_chunk,
                /*column =*/ scores_max_local,
                /*ldm =*/ scores_chunk_ldm,
                chunk_size_q_real,
                chunk_size_k_real
            );
        }
        
        __syncthreads();

        if (kv_row_chunk != 0) {
            // denom_local = scores_chunk.sum(dim=1)
            threadblock_row_sum<chunk_size_q, chunk_size_k, n_warps, /*copy_to_vec=*/true>(
                /*mat = */ scores_chunk,
                /*vec = */ denom_local,
                /*aux_memory = */ aux_mem,
                /*ldm = */ scores_chunk_ldm,
                /*ldm_aux = */ reduce_sum_ldm,
                chunk_size_q_real,
                chunk_size_k_real
            );
        }
            
        __syncthreads();

        if (kv_row_chunk != 0) {
            // numer_local = scores_chunk @ values_chunk
            threadblock_gemm_k_real< /*m=*/chunk_size_q, /*n=*/head_dim, /*k=*/chunk_size_k, n_warps, 
                            /*transpose_b=*/false, /*alpha_is_one=*/true, /*preloaded_a_frags=*/false >(
                /*mat_a_frags = */ nullptr,
                /*mat_a = */ scores_chunk,
                /*mat_b = */ &values[batch_offset_k + kv_row * head_dim],
                /*mat_c = */ numer_local,
                /*m_real = */ chunk_size_q_real,
                /*n_real = */ head_dim,
                /*k_real = */ chunk_size_k_real,
                /*lda = */ scores_chunk_ldm,
                /*ldb = */ head_dim,
                /*ldc = */ numer_local_ldm
            );
        }
        
        __syncthreads();
        
        if (kv_row_chunk != 0) {
            threadblock_aggregate_softmax<chunk_size_k, head_dim, n_warps>(
                /*numer_a = */ numer,
                /*denom_a = */ denom,
                /*max_a = */ scores_max,
                /*numer_b = */ numer_local,
                /*denom_b = */ denom_local,
                /*max_b = */ scores_max_local,
                /*lda =*/ numer_ldm,
                /*ldb =*/ numer_local_ldm,
                /*aux =*/ aux_mem,
                chunk_size_k_real
            );
        }
    }

    __syncthreads();
    
    // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {

    //     printf("numer: \n");
    //     for (int i = 0; i < chunk_size_q; i++) {
    //         for (int j = 0; j < head_dim; j++) {
    //             printf("%.1f ", __half2float(numer[i * numer_ldm + j]));
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {

    //     printf("denom: \n");
    //     for (int i = 0; i < chunk_size_q; i++) {
    //         printf("%.1f ", __half2float(denom[i]));
    //     }
    //     printf("\n");
    // }
    threadblock_divide_and_store_chunk<chunk_size_q, head_dim, n_warps>(
        /*numer = */ numer,
        /*denom = */ denom,
        /*dst = */ &output[batch_offset_q_no_trans + q_row * num_features + head_col],
        /*lds = */ numer_ldm,
        /*ldd = */ num_features,
        /*seq_len*/chunk_size_q_real
    );

    // threadblock_divide_and_store_chunk<chunk_size, head_dim, n_warps>(
    //     /*numer = */ numer,
    //     /*denom = */ denom,
    //     /*dst = */ &output[batch_offset_q_no_trans + head_col],
    //     /*lds = */ numer_ldm,
    //     /*ldd = */ num_features,
    //     /*seq_len*/chunk_size_q_real
    // );
}

template <uint32_t head_dim, uint32_t chunk_size>
void launch_attention_trans_kernel(
    uint32_t batch_size,
    uint32_t seq_len_q,
    uint32_t seq_len_k,
    uint32_t num_features,
    const half_t* queries,
    const half_t* keys,
    const half_t* values,
    const half_t* mask,
    half_t* output,
    bool syncronize = false
) {
    if (seq_len_k < 24) {
        constexpr uint32_t n_warps = chunk_size < head_dim ? chunk_size / 16 : head_dim / 16;
        constexpr uint32_t max_size = chunk_size > head_dim ? chunk_size : head_dim;
        constexpr uint32_t shared_mem_size = ( chunk_size * (head_dim + 8) + (chunk_size + 16) * (max_size + 8) + 
                                            16 * (chunk_size + 8) + 4 * chunk_size ) * sizeof(half_t);
        if (warp_size * n_warps >= 1024) {
            printf("kernel thread in block is greater than 1024!\n");
            return;
        }
        // Call attention kernel
        dim3 threads(warp_size, n_warps, 1);
        dim3 blocks(batch_size * num_features / head_dim, 1, 1);
        
        attention_trans_kernel_64<head_dim, chunk_size, n_warps><<<dim3(blocks), dim3(threads), shared_mem_size>>>(
            batch_size, seq_len_q, seq_len_k, num_features, 
            queries, keys, values, mask, output
        );
    } else {
        constexpr uint32_t n_warps = 4;
        constexpr uint32_t max_size = chunk_size > head_dim ? chunk_size : head_dim;
        constexpr uint32_t seq_len_k_tiling_size = 128;
        constexpr uint32_t chunk_size_k = 128;

        constexpr uint32_t queries_chunk_size = chunk_size * (head_dim + 8);
        constexpr uint32_t scores_chunk_size = (chunk_size + 16) * (seq_len_k_tiling_size + 8);
        constexpr uint32_t aux_size = 16 * (seq_len_k_tiling_size + 8);
        constexpr uint32_t scores_max_local_size = seq_len_k_tiling_size;
        constexpr uint32_t denom_size = seq_len_k_tiling_size;
        constexpr uint32_t denom_local_size = seq_len_k_tiling_size;
        constexpr uint32_t scores_max_size = seq_len_k_tiling_size;


        constexpr uint32_t shared_mem_size = (queries_chunk_size + scores_chunk_size + aux_size + 
                                            scores_max_local_size + denom_size + denom_local_size + scores_max_size) * sizeof(half_t);
        if (warp_size * n_warps >= 1024) {
            printf("kernel thread in block is greater than 1024!\n");
            return;
        }
        // Call attention kernel
        dim3 threads(warp_size, n_warps, 1);
        dim3 blocks(batch_size * num_features / head_dim, 1, 1);
        
        attention_trans_kernel_256<head_dim, chunk_size, chunk_size_k, n_warps><<<dim3(blocks), dim3(threads), shared_mem_size>>>(
            batch_size, seq_len_q, seq_len_k, num_features, 
            queries, keys, values, mask, output
        );
    }

    hipDeviceSynchronize();

    if (syncronize) {
        CHECK_CUDA_ERROR(hipDeviceSynchronize());
        CHECK_LAST_CUDA_ERROR();        
    }
}