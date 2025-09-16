// nvcc -O3 --use_fast_math -std=c++17 -gencode arch=compute_86,code=sm_86 -o matmul matmul_3.cu
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

__device__ __forceinline__ void cp_async4(void *smem_ptr, const void *glob_ptr) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;" ::"r"(smem),
        "l"(glob_ptr),
        "n"(BYTES));
}

__device__ __forceinline__ void async_memcpy_waitall() {
    asm volatile("cp.async.wait_all;\n" ::);
}

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Too slow to actually run!)
//
// void matmul_cpu_naive(
//     int32_t size_i,
//     int32_t size_j,
//     int32_t size_k,
//     float const *a,
//     float const *b,
//     float *c) {
//     for (int32_t i = 0; i < size_i; ++i) {
//         for (int32_t j = 0; j < size_j; ++j) {
//             float sum = 0.0;
//             for (int32_t k = 0; k < size_k; ++k) {
//                 sum += a[i * size_k + k] * b[k * size_j + j];
//             }
//             c[i * size_j + j] = sum;
//         }
//     }
// }

/// <--- your code here --->

/*
    // OPTIONAL: Uncomment this block to include your kernel implementation
    // from Lab 5 for easy comparison.

    ////////////////////////////////////////////////////////////////////////////////
    // Optimized GPU Implementation with Reduction along k (Baseline from Lab 5)

    #define HAS_LAB_5_BASELINE_IMPL // <~~ keep this line if you want to benchmark your Lab 5 kernel!

    namespace matmul_improved_reduce {

    // TODO: your GPU kernels here...

    size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        // TODO: your CPU code here
        return 0;
    }

    void launch_matmul_improved_reduce(
        int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a, // pointer to GPU memory
        float const *b, // pointer to GPU memory
        float *c,       // pointer to GPU memory
        void *workspace // pointer to GPU memory
    ) {
        // TODO: your CPU code here
    }

    } // namespace matmul_improved_reduce
*/

////////////////////////////////////////////////////////////////////////////////
// Tensor Core GPU Implementation

__device__ __forceinline__ void swap(float* &a, float* &b) {
    float* temp = a;
    a = b;
    b = temp;
}

namespace matmul_tensor {

constexpr int REDUCE_DIM = 32;

__device__ __forceinline__ void load_tiles_to_shared_fp4(
    float* a_shared, 
    float* b_shared, 
    const float* a, 
    const float* b, 
    int size_i,
    int size_j,
    int size_k,
    int tile_idx,
    dim3 tileDim,
    int threadIdx_lin) {

    const int threadIdx_lin4 = 4 * threadIdx_lin;

    const int y_a = tileDim.y*blockIdx.y + (threadIdx_lin4 / REDUCE_DIM);
    const int x_a = tile_idx*REDUCE_DIM + (threadIdx_lin4 % REDUCE_DIM);

    const int yb_local = threadIdx_lin4 / tileDim.x;
    const int y_b = tile_idx*REDUCE_DIM + (yb_local);
    const int x_b = tileDim.x*blockIdx.x + (threadIdx_lin4 % tileDim.x);

    void* a_shmem_addr = a_shared + threadIdx_lin4;
    void* b_shmem_addr = b_shared + threadIdx_lin4;

    if (y_a < size_i && x_a < size_k) {
        const void* a_gmem_addr = a + ((y_a)*size_k + (x_a));
        cp_async4(a_shmem_addr, a_gmem_addr);
    } else {    
        reinterpret_cast<float4*>(a_shmem_addr)[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // for now disable threads for tile b that load k values of 32 or greater
    if (yb_local >= 32) return;

    if (y_b < size_k && x_b < size_j) {
        const void* b_gmem_addr = b + ((y_b)*size_j + (x_b));
        cp_async4(b_shmem_addr, b_gmem_addr);
    } else {
        reinterpret_cast<float4*>(b_shmem_addr)[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

__device__ __forceinline__ void mma_16x8x8(float const *a, float const *b, int* d0, int* d1, int* d2, int* d3, dim3 tileDim, int threadIdx_lin_local) {
    const int a0 = (threadIdx_lin_local % 4) + REDUCE_DIM * (threadIdx_lin_local / 4);
    const int a1 = a0 + 8 * REDUCE_DIM;
    const int a2 = a0 + 4;
    const int a3 = a1 + 4;

    const int b0 = (threadIdx_lin_local % 4) * tileDim.x + (threadIdx_lin_local / 4);
    const int b1 = b0 + 4 * tileDim.x;

    // TODO: is ASM correct
    asm(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 \
        {%0, %1, %2, %3},     /* 'D' matrix */ \
        {%4, %5, %6, %7},     /* 'A' matrix */ \
        {%8, %9},             /* 'B' matrix */ \
        {%0, %1, %2, %3}      /* 'C' matrix - Same as D  */;"
        : "+r"(*d0), "+r"(*d1), "+r"(*d2), "+r"(*d3)
        : "r"(__float_as_uint(a[a0])), "r"(__float_as_uint(a[a1])), "r"(__float_as_uint(a[a2])), "r"(__float_as_uint(a[a3])),
          "r"(__float_as_uint(b[b0])), "r"(__float_as_uint(b[b1]))
    );
}

__device__ __forceinline__ void reduce(
    const float* a_shmem, 
    const float* b_shmem, 
    int* d0, 
    int* d1, 
    int* d2, 
    int* d3, 
    int x_idx_start,
    int y_idx_start,
    dim3 tileDim, 
    int threadIdx_lin_local) {

    const float* a_start = a_shmem + (y_idx_start * REDUCE_DIM);
    const float* b_start = b_shmem + x_idx_start;

    const int num_tiles = REDUCE_DIM / 8;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        mma_16x8x8(a_start, b_start, d0, d1, d2, d3, tileDim, threadIdx_lin_local);
        a_start += 8;
        b_start += 8 * tileDim.x;
    }
}


__global__ void matmul_tensor(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *c /* pointer to GPU memory */) {

    const dim3 tileDim{blockDim.x, 4 * blockDim.y};

    // shared memory setup
    extern __shared__ float shmem[];
    const int a_tile_size = tileDim.y * REDUCE_DIM;
    const int b_tile_size = REDUCE_DIM * tileDim.x;
    float* a_shmem0 = shmem;              // (tileDim.y x REDUCE_DIM)
    float* b_shmem0 = &shmem[a_tile_size];  // (REDUCE_DIM x tileDim.x)

    float* a_shmem1 = &shmem[a_tile_size + b_tile_size];
    float* b_shmem1 = &shmem[2 * a_tile_size + b_tile_size];

    // warp idxs setup
    const int threadIdx_lin = (threadIdx.y * blockDim.x) + threadIdx.x; // linearize
    const int threadIdx_lin_local = threadIdx_lin % 32;
    const int warp_idx = threadIdx_lin / 32;
    const int y_idx_start = 16 * (warp_idx / 4); 
    const int x_idx_start = 8 * (warp_idx % 4);

    // partial sums
    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    // load inital tiles
    load_tiles_to_shared_fp4(a_shmem0, b_shmem0, a, b, size_i, size_j, size_k, 0, tileDim, threadIdx_lin);
    async_memcpy_waitall();
    __syncthreads();

    uint32_t tile_iters = (size_k + REDUCE_DIM - 1) / REDUCE_DIM;
    for (int tile_idx = 1; tile_idx < tile_iters; tile_idx++) {
        // load in next tile
        load_tiles_to_shared_fp4(a_shmem1, b_shmem1, a, b, size_i, size_j, size_k, tile_idx, tileDim, threadIdx_lin);

        // reduce(t, tileDim, a_shared0, b_shared0, sum);
        reduce(a_shmem0, b_shmem0, &d0, &d1, &d2, &d3, x_idx_start, y_idx_start, tileDim, threadIdx_lin_local);

        swap(a_shmem0, a_shmem1);
        swap(b_shmem0, b_shmem1);

        async_memcpy_waitall();
        __syncthreads();
    }

    // last last tile
    reduce(a_shmem0, b_shmem0, &d0, &d1, &d2, &d3, x_idx_start, y_idx_start, tileDim, threadIdx_lin_local);
    
    // store sum
    float* c_start = c + (blockIdx.y * tileDim.y * size_j) + (blockIdx.x * tileDim.x);
    c_start += (y_idx_start * size_j) + x_idx_start;

    const int c0 = 2 * (threadIdx_lin_local % 4) + size_j * (threadIdx_lin_local / 4);
    const int c1 = c0 + 1;
    const int c2 = c0 + 8 * size_j;
    const int c3 = c1 + 8 * size_j;

    c_start[c0] = __uint_as_float(d0);
    c_start[c1] = __uint_as_float(d1);
    c_start[c2] = __uint_as_float(d2);
    c_start[c3] = __uint_as_float(d3);
}

size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
    /* TODO: your CPU code here */
    return 0;
}

void launch_matmul_tensor(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *c,       /* pointer to GPU memory */
    void *workspace /* pointer to GPU memory */
) {
    /* TODO: your CPU code here */
    dim3 block_size(32, 32);
    dim3 tile_size(block_size.x, 4 * block_size.y); // does block_size * k work?
    dim3 num_blocks((size_j + tile_size.x - 1) / tile_size.x,
                    (size_i + tile_size.y - 1) / tile_size.y);
    
    // uint32_t shmem_bytes = 2 * 2 * tile_size.x * REDUCE_DIM * sizeof(float);
    uint32_t shmem_bytes = 2 * ((tile_size.y * REDUCE_DIM) + (REDUCE_DIM * tile_size.x)) * sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute(
        matmul_tensor,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        shmem_bytes));
    
    matmul_tensor<<<num_blocks, block_size, shmem_bytes>>>(size_i, size_j, size_k, a, b, c);

    CUDA_CHECK(cudaGetLastError());
}

}; // namespace matmul_tensor

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

std::vector<float> read_data(std::string const &path, int32_t size) {
    std::ifstream file(path, std::ios::binary);
    std::vector<float> data(size);
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
    if (file.fail()) {
        std::cerr << "Failed to read " << path << std::endl;
        std::abort();
    }
    return data;
}

template <typename Reset, typename F>
double
benchmark_ms(double target_time_ms, int32_t num_iters_inner, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        for (int32_t i = 0; i < num_iters_inner; ++i) {
            f();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms / num_iters_inner);
    }
    return best_time_ms;
}

struct BenchmarkConfig {
    int32_t size_i;
    int32_t size_j;
    int32_t size_k;
};

struct TestData {
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> a;
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> b;
    std::map<std::tuple<int32_t, int32_t, int32_t>, std::vector<float>> c;
};

TestData read_test_data(
    std::string const &test_data_dir,
    std::vector<BenchmarkConfig> const &configs) {
    auto data = TestData{};
    for (auto const &config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto size_k = config.size_k;

        auto path_prefix = test_data_dir + "/test_";

        if (data.a.find({size_i, size_k}) == data.a.end()) {
            data.a[{size_i, size_k}] = read_data(
                path_prefix + "a_" + std::to_string(size_i) + "x" +
                    std::to_string(size_k) + ".bin",
                size_i * size_k);
        }

        if (data.b.find({size_k, size_j}) == data.b.end()) {
            data.b[{size_k, size_j}] = read_data(
                path_prefix + "b_" + std::to_string(size_k) + "x" +
                    std::to_string(size_j) + ".bin",
                size_k * size_j);
        }

        if (data.c.find({size_i, size_j, size_k}) == data.c.end()) {
            data.c[{size_i, size_j, size_k}] = read_data(
                path_prefix + "c_" + std::to_string(size_i) + "x" +
                    std::to_string(size_j) + "x" + std::to_string(size_k) + ".bin",
                size_i * size_j);
        }
    }
    return data;
}

struct BenchmarkResults {
    char const *name;
    std::map<std::tuple<int32_t, int32_t, int32_t>, double> elapsed_ms;
};

enum class Phase {
    WARMUP,
    BENCHMARK,
};

template <typename Impl>
void run_config(
    Phase phase,
    TestData const &data,
    BenchmarkConfig const &config,
    BenchmarkResults &results) {
    auto size_i = config.size_i;
    auto size_j = config.size_j;
    auto size_k = config.size_k;

    auto const &a = data.a.at({size_i, size_k});
    auto const &b = data.b.at({size_k, size_j});
    auto const &c = data.c.at({size_i, size_j, size_k});

    float *a_gpu;
    float *b_gpu;
    float *c_gpu;
    CUDA_CHECK(cudaMalloc(&a_gpu, size_i * size_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_gpu, size_k * size_j * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c_gpu, size_i * size_j * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        a_gpu,
        a.data(),
        size_i * size_k * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        b_gpu,
        b.data(),
        size_k * size_j * sizeof(float),
        cudaMemcpyHostToDevice));

    size_t workspace_size = Impl::get_workspace_size(size_i, size_j, size_k);
    void *workspace_gpu = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_gpu, workspace_size));
        CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
    }

    if (phase == Phase::BENCHMARK) {
        printf("  %6d  %6d  %6d", size_i, size_j, size_k);
    } else {
        printf("  warmup %6d  %6d  %6d", size_i, size_j, size_k);
    }

    Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu, workspace_gpu);

    std::vector<float> c_out_host(size_i * size_j);
    CUDA_CHECK(cudaMemcpy(
        c_out_host.data(),
        c_gpu,
        size_i * size_j * sizeof(float),
        cudaMemcpyDeviceToHost));

    double mse = 0.0;
    double ref_mean_square = 0.0;
    for (int32_t i = 0; i < size_i; ++i) {
        for (int32_t j = 0; j < size_j; ++j) {
            float diff = c_out_host[i * size_j + j] - c[i * size_j + j];
            mse += diff * diff;
            ref_mean_square += c[i * size_j + j] * c[i * size_j + j];
        }
    }
    mse /= size_i * size_j;
    ref_mean_square /= size_i * size_j;
    float rmse = std::sqrt(mse);
    float rel_rmse = rmse / std::sqrt(ref_mean_square);

    if (phase == Phase::BENCHMARK) {
        printf("  %8.02e", rel_rmse);
    }

    if (rel_rmse > 1e-3) {
        if (phase == Phase::BENCHMARK) {
            printf("  %9s  %7s", "-", "-");
        }
    } else {
        double target_time_ms = 200.0;
        double elapsed_ms = benchmark_ms(
            target_time_ms,
            4,
            [&]() {
                if (workspace_size > 0) {
                    CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
                }
            },
            [&]() {
                Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu, workspace_gpu);
            });

        if (phase == Phase::BENCHMARK) {
            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            printf("  %9.02f  %7.02f", elapsed_ms, tflop / (elapsed_ms * 1e-3));

            results.elapsed_ms[{size_i, size_j, size_k}] = elapsed_ms;
        }
    }

    printf("\n");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaFree(workspace_gpu));
    }
}

template <typename Impl>
BenchmarkResults run_all_configs(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = BenchmarkResults{Impl::name};
    if (phase == Phase::WARMUP) {
        printf("warmup %s:\n\n", Impl::name);
    } else {
        printf("%s:\n\n", Impl::name);
        printf(
            "  %-6s  %-6s  %-6s  %-8s  %-9s  %-7s\n",
            "size_i",
            "size_j",
            "size_k",
            "RRMSE",
            "time (ms)",
            "TFLOP/s");
        printf(
            "  %-6s  %-6s  %-6s  %-8s  %-9s  %-7s\n",
            "------",
            "------",
            "------",
            "--------",
            "---------",
            "-------");
    }
    for (auto const &config : configs) {
        run_config<Impl>(phase, data, config, results);
    }
    printf("\n");
    return results;
}

#ifdef HAS_LAB_5_BASELINE_IMPL

struct MatmulImprovedReduce {
    constexpr static char const *name = "matmul_improved_reduce";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return matmul_improved_reduce::get_workspace_size(size_i, size_j, size_k);
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_improved_reduce::launch_matmul_improved_reduce(
            size_i,
            size_j,
            size_k,
            a,
            b,
            c,
            workspace);
    }
};

#endif

struct MatmulTensor {
    constexpr static char const *name = "matmul_tensor";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return matmul_tensor::get_workspace_size(size_i, size_j, size_k);
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_tensor::launch_matmul_tensor(size_i, size_j, size_k, a, b, c, workspace);
    }
};

BenchmarkResults get_cublas_fma_results() {
    // Hard-coded data collected on A4000 GPU
    return BenchmarkResults{
        "cublas_fma",
        {
            {{3072, 3072, 3072}, 4.05},
            {{512, 3072, 3072}, 0.80},
            {{256, 3072, 3072}, 0.46},
            {{128, 3072, 3072}, 0.24},
            {{64, 3072, 3072}, 0.13},
            {{32, 3072, 3072}, 0.11},
            {{16, 3072, 3072}, 0.11},
        }};
}

std::vector<BenchmarkResults> run_all_impls(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = std::vector<BenchmarkResults>{};
#ifdef HAS_LAB_5_BASELINE_IMPL
    results.push_back(run_all_configs<MatmulImprovedReduce>(phase, data, configs));
#endif
    results.push_back(run_all_configs<MatmulTensor>(phase, data, configs));
    return results;
}

void write_json_results(
    std::string const &path,
    std::vector<BenchmarkResults> const &results) {
    auto file = std::ofstream(path);
    file << "{\n";
    for (int32_t i = 0; i < results.size(); ++i) {
        auto const &result = results.at(i);
        file << "  \"" << result.name << "\": [\n";
        int32_t j = 0;
        for (auto const &[config, elapsed_ms] : result.elapsed_ms) {
            auto [size_i, size_j, size_k] = config;
            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            double tflop_per_sec = tflop / (elapsed_ms * 1e-3);
            file << "    {\n";
            file << "      \"size_i\": " << size_i << ",\n";
            file << "      \"size_j\": " << size_j << ",\n";
            file << "      \"size_k\": " << size_k << ",\n";
            file << "      \"elapsed_ms\": " << elapsed_ms << ",\n";
            file << "      \"tflop_per_sec\": " << tflop_per_sec << "\n";
            file << "    }";
            if (j + 1 < result.elapsed_ms.size()) {
                file << ",";
            }
            file << "\n";
            ++j;
        }
        file << "  ]";
        if (i + 1 < results.size()) {
            file << ",";
        }
        file << "\n";
    }
    file << "}\n";
}

void print_speedup(
    std::vector<BenchmarkConfig> const &configs,
    BenchmarkResults const &first,
    BenchmarkResults const &second) {
    printf("\nspeedups %s -> %s:\n\n", first.name, second.name);
    printf("  %-6s  %-6s  %-6s  %-7s\n", "size_i", "size_j", "size_k", "speedup");
    printf("  %-6s  %-6s  %-6s  %-7s\n", "------", "------", "------", "-------");
    for (auto const &config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto size_k = config.size_k;
        printf("  %6d  %6d  %6d", size_i, size_j, size_k);
        auto it_first = first.elapsed_ms.find({size_i, size_j, size_k});
        auto it_second = second.elapsed_ms.find({size_i, size_j, size_k});
        if (it_first != first.elapsed_ms.end() && it_second != second.elapsed_ms.end()) {
            printf("  %6.02fx", it_first->second / it_second->second);
        } else {
            printf("  %7s", "-");
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    std::string test_data_dir = ".";
    if (char *c_str_test_data_dir = std::getenv("MATMUL_TEST_DATA_DIR_2")) {
        test_data_dir = c_str_test_data_dir;
    }

    auto configs = std::vector<BenchmarkConfig>{
        {3072, 3072, 3072},
        // {512, 3072, 3072},
        // {256, 3072, 3072},
        // {128, 3072, 3072},
        // {64, 3072, 3072},
        // {32, 3072, 3072},
        // {16, 3072, 3072},
    };
    auto data = read_test_data(test_data_dir, configs);
    run_all_impls(Phase::WARMUP, data, configs);
    auto results = run_all_impls(Phase::BENCHMARK, data, configs);

    for (int32_t j = 1; j < results.size(); ++j) {
        for (int32_t i = j; i > 0;) {
            --i;
            print_speedup(configs, results.at(i), results.at(j));
        }
    }

    printf("\n-----------------------------------------------------------\n");
    printf("---- Comparison to non-tensor-core cuBLAS performance: ----\n");
    printf("-----------------------------------------------------------\n");

    print_speedup(configs, get_cublas_fma_results(), results.at(results.size() - 1));

    write_json_results("out/results.json", results);

    return 0;
}