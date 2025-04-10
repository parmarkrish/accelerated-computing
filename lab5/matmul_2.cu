// nvcc -O3 --use_fast_math -std=c++17 -gencode arch=compute_86,code=sm_86 -o matmul matmul_2.cu
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
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

__device__ inline void cp_async4(void *smem_ptr, const void *glob_ptr) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" ::"r"(smem),
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

// OPTIONAL: Uncomment this block to include your kernel implementation
// from Lab 4 for easy comparison.

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation with Reuse in L1/Shmem and Registers (Baseline from Lab 4)

#define HAS_LAB_4_BASELINE_IMPL // <~~ keep this line if you want to benchmark your Lab 4 kernel!

namespace matmul_l1_reg {

// K defines the microtile width and height
#define K 2 

__global__ void matmul_l1_reg(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    dim3 tileDim{K * blockDim.x, K * blockDim.y};
    uint2 t{K * threadIdx.x, K * threadIdx.y};
    uint32_t i = tileDim.y * blockIdx.y + t.y;
    uint32_t j = tileDim.x * blockIdx.x + t.x;

    extern __shared__ float shmem[];
    float* a_shared = shmem;
    float* b_shared = &shmem[tileDim.x * tileDim.y];
    
    if (i >= size_i || j >= size_j) return;

    float a_microtile[K][K] = {0};
    float b_microtile[K][K] = {0};
    
    float sum[K][K] = {0};

    uint32_t tile_iters = (size_k + tileDim.x - 1) / tileDim.x;
    for (int tile_idx = 0; tile_idx < tile_iters; tile_idx++) {
        // load into shared memory
        #pragma unroll
        for (int y = 0; y < K; y++) {
            #pragma unroll
            for (int x = 0; x < K; x++) {
                a_shared[(t.y+y)*tileDim.x + (t.x+x)] = a[(i+y)*size_k + (tile_idx*tileDim.x + t.x+x)];
                b_shared[(t.y+y)*tileDim.x + (t.x+x)] = b[(tile_idx * tileDim.y + t.y+y)*size_j + (j+x)];
            }
        }
        __syncthreads();
        for (int microtile_idx = 0; microtile_idx < tileDim.x/K; microtile_idx++) {
            // load into microtile
            #pragma unroll
            for (int y = 0; y < K; y++) {
                #pragma unroll
                for (int x = 0; x < K; x++) {
                    a_microtile[y][x] = a_shared[(t.y+y)*tileDim.x + (microtile_idx*K + x)];
                    b_microtile[y][x] = b_shared[(microtile_idx * K + y)*tileDim.x + (t.x+x)];
                }
            }

            // compute microtile
            #pragma unroll
            for (int y = 0; y < K; y++) {
                #pragma unroll
                for (int x = 0; x < K; x++) {
                    #pragma unroll
                    for (int k = 0; k < K; k++) {
                        sum[y][x] += a_microtile[y][k] * b_microtile[k][x];
                    }
                }
            }
        }
        __syncthreads();
    }
    // store microtile sums
    #pragma unroll
    for (int y = 0; y < K; y++) {
        #pragma unroll
        for (int x = 0; x < K; x++) {
            c[(i+y)*size_j + (j+x)] = sum[y][x];
        }
    }
}

void launch_matmul_l1_reg(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    dim3 block_size(32, 32);
    dim3 tile_size(K * block_size.x, K * block_size.y); // does block_size * k work?
    dim3 num_blocks((size_j + tile_size.x - 1) / tile_size.x,
                    (size_i + tile_size.y - 1) / tile_size.y);
    
    uint32_t shmem_bytes = 2 * tile_size.x * tile_size.y * sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute(
        matmul_l1_reg,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        shmem_bytes));
    
    matmul_l1_reg<<<num_blocks, block_size, shmem_bytes>>>(size_i, size_j, size_k, a, b, c);
}

} // namespace matmul_l1_reg

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

namespace matmul_improved {

#define K2 3 
#define R 2

__global__ void matmul_improved(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *c /* pointer to GPU memory */) {
    
    const dim3 tileDim{K2 * blockDim.x, K2 * blockDim.y};
    uint32_t i = tileDim.y * blockIdx.y + threadIdx.y;
    uint32_t j = tileDim.x * blockIdx.x + threadIdx.x;

    extern __shared__ float shmem[];
    float* a_shared = shmem;
    float* b_shared = &shmem[tileDim.x * tileDim.y];
    
    // if (i >= size_i || j >= size_j) return;

    float a_microtile[K2] = {0};
    float b_microtile[K2] = {0};
    
    float sum[K2][K2] = {0};

    uint32_t tile_iters = (size_k + tileDim.x - 1) / tileDim.x;
    for (int tile_idx = 0; tile_idx < tile_iters; tile_idx++) {
        // load into shared memory
        #pragma unroll
        for (int y = 0; y < K2; y++) {
            #pragma unroll
            for (int x = 0; x < K2; x++) {
                uint32_t ty = threadIdx.y + y*blockDim.y;
                uint32_t tx = threadIdx.x + x*blockDim.x;

                a_shared[ty*tileDim.x + tx] = a[(i+(y*blockDim.y))*size_k + (tile_idx*tileDim.x + tx)];
                b_shared[ty*tileDim.x + tx] = b[(tile_idx*tileDim.y + ty)*size_j + (j+(x*blockDim.x))];
            }
        }
        __syncthreads();

        for (int k = 0; k < tileDim.y; k++) {
            // load into microtile
            #pragma unroll
            for (int i = 0; i < K2; i++) {
                a_microtile[i] = a_shared[(i*blockDim.y + threadIdx.y)*tileDim.x + (k)];
                b_microtile[i] = b_shared[(k)*tileDim.x + (threadIdx.x + i*blockDim.x)];
            }

            // compute microtile
            #pragma unroll
            for (int y = 0; y < K2; y++) {
                #pragma unroll
                for (int x = 0; x < K2; x++) {
                    sum[y][x] += a_microtile[y] * b_microtile[x];
                }
            }
        }
        __syncthreads();
    }
    // store microtile sums
    #pragma unroll
    for (int y = 0; y < K2; y++) {
        #pragma unroll
        for (int x = 0; x < K2; x++) {
            c[(i + y*blockDim.y)*size_j + (j + x*blockDim.x)] = sum[y][x];
        }
    }
}

void launch_matmul_improved(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *c /* pointer to GPU memory */) {
    
    dim3 block_size(32, 32);
    dim3 tile_size(K2 * block_size.x, K2 * block_size.y); // does block_size * k work?
    dim3 num_blocks((size_j + tile_size.x - 1) / tile_size.x,
                    (size_i + tile_size.y - 1) / tile_size.y);
    
    uint32_t shmem_bytes = 2 * tile_size.x * tile_size.y * sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute(
        matmul_improved,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        shmem_bytes));
    
    matmul_improved<<<num_blocks, block_size, shmem_bytes>>>(size_i, size_j, size_k, a, b, c);
}

}; // namespace matmul_improved

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation with Reduction along k

namespace matmul_improved_reduce {

/* TODO: your GPU kernels here... */

size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
    /* TODO: your CPU code here */
    return 0;
}

void launch_matmul_improved_reduce(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *c,       /* pointer to GPU memory */
    void *workspace /* pointer to GPU memory */
) {
    /* TODO: your CPU code here */
}

}; // namespace matmul_improved_reduce

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

    if (rel_rmse > 1e-5) {
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

#ifdef HAS_LAB_4_BASELINE_IMPL

struct MatmulL1Reg {
    constexpr static char const *name = "matmul_l1_reg";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return 0;
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_l1_reg::launch_matmul_l1_reg(size_i, size_j, size_k, a, b, c);
    }
};

#endif

struct MatmulImproved {
    constexpr static char const *name = "matmul_improved";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return 0;
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_improved::launch_matmul_improved(size_i, size_j, size_k, a, b, c);
    }
};

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

std::vector<BenchmarkResults> run_all_impls(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = std::vector<BenchmarkResults>{};
#ifdef HAS_LAB_4_BASELINE_IMPL
    results.push_back(run_all_configs<MatmulL1Reg>(phase, data, configs));
#endif
    results.push_back(run_all_configs<MatmulImproved>(phase, data, configs));
    results.push_back(run_all_configs<MatmulImprovedReduce>(phase, data, configs));
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
        // {1, 3072, 3072},
        // {256, 256, 256},
        // {256, 256, 1024},
        // {256, 256, 8192},
        // {128, 128, 32768},
    };
    auto data = read_test_data(test_data_dir, configs);
    run_all_impls(Phase::WARMUP, data, configs);
    auto results = run_all_impls(Phase::BENCHMARK, data, configs);

    for (int32_t j = 1; j < results.size(); ++j) {
        for (int32_t i = j; i > 0;) {
            --i;
            auto const &first = results.at(i);
            auto const &second = results.at(j);
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
                if (it_first != first.elapsed_ms.end() &&
                    it_second != second.elapsed_ms.end()) {
                    printf("  %6.02fx", it_first->second / it_second->second);
                } else {
                    printf("  %7s", "-");
                }
                printf("\n");
            }
        }
    }

    write_json_results("out/results.json", results);

    return 0;
}