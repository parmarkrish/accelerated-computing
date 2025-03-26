// nvcc -O3 --use_fast_math -gencode arch=compute_86,code=sm_86 -o matmul matmul.cu
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
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

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation (With Reuse in L1/Shmem)

namespace matmul_l1 {

__global__ void matmul_l1(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    extern __shared__ float shmem[];
    float* a_shared = shmem;
    float* b_shared = &shmem[blockDim.x * blockDim.y];

    uint32_t i = blockDim.y * blockIdx.y + threadIdx.y;
    uint32_t j = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i >= size_i || j >= size_j) return;

    uint32_t tile_iters = (size_k + blockDim.x - 1) / blockDim.x;

    uint32_t idx_local = threadIdx.y * blockDim.x + threadIdx.x;
    float sum = 0;
    for (int tile_idx = 0; tile_idx < tile_iters; tile_idx++) {
        uint32_t j_tile = tile_idx*blockDim.x + threadIdx.x;
        if (j_tile < size_j) {
            a_shared[idx_local] = a[i*size_k + j_tile];
        } else {
            a_shared[idx_local] = 0;
        }

        uint32_t i_tile = tile_idx * blockDim.y + threadIdx.y;
        if (i_tile < size_i) {
            b_shared[idx_local] = b[i_tile*size_j + j];
        } else {
            b_shared[idx_local] = 0;
        }
        __syncthreads();

        for (int k = 0; k < blockDim.x; k++) {
            sum += a_shared[threadIdx.y * blockDim.x + k] * b_shared[k * blockDim.x + threadIdx.x];
        }

        __syncthreads();
    }

    c[i * size_j + j] = sum;
}

void launch_matmul_l1(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    dim3 block_size(32, 32);
    dim3 num_blocks((size_j + block_size.x - 1) / block_size.x,
                    (size_i + block_size.y - 1) / block_size.y);
    
    uint32_t shmem_bytes = 2 * block_size.x * block_size.y * sizeof(float);
    
    matmul_l1<<<num_blocks, block_size, shmem_bytes>>>(size_i, size_j, size_k, a, b, c);
}

}; // namespace matmul_l1

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation (With Reuse in L1/Shmem and Registers)

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

}; // namespace matmul_l1_reg

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

template <typename F>
double benchmark_ms(double target_time_ms, int32_t num_iters_inner, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
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

struct BenchmarkResult {
    char const *name;
    double elapsed_ms;
};

struct BenchmarkConfig {
    int32_t size_i;
    int32_t size_j;
    int32_t size_k;
    bool save_result;
};

template <typename Impl>
void run_tests_for_size(
    std::string const &test_data_dir,
    std::vector<BenchmarkResult> &saved_results,
    std::vector<BenchmarkConfig> const &configs) {
    for (auto config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto size_k = config.size_k;

        auto path_prefix = test_data_dir + "/test_" + std::to_string(size_i) + "x" +
            std::to_string(size_j) + "x" + std::to_string(size_k);
        auto a = read_data(path_prefix + "_a.bin", size_i * size_k);
        auto b = read_data(path_prefix + "_b.bin", size_k * size_j);
        auto c = read_data(path_prefix + "_c.bin", size_i * size_j);

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

        Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu);

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

        printf("  size %4d * %4d * %4d:\n", size_i, size_j, size_k);
        printf("    correctness: %.02e relative RMSE\n", rel_rmse);

        if (rel_rmse > 1e-5) {
            printf("    skipping benchmark (incorrect)\n");
        } else {
            double elapsed_ms = benchmark_ms(1000.0, 4, [&]() {
                Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu);
            });

            printf("    run time: %6.02f ms\n", elapsed_ms);

            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            printf("    throughput: %5.02f TFLOP/s\n", tflop / (elapsed_ms * 1e-3));

            if (config.save_result) {
                saved_results.push_back({Impl::name, elapsed_ms});
            }
        }

        printf("\n");
    }
}

template <typename Impl>
void run_all_tests(
    std::string const &test_data_dir,
    std::vector<BenchmarkResult> &saved_results) {
    printf("%s:\n\n", Impl::name);
    run_tests_for_size<Impl>(test_data_dir, saved_results, {{256, 256, 256, false}});
    run_tests_for_size<Impl>(test_data_dir, saved_results, {{3072, 3072, 3072, true}});
}

struct MatmulL1 {
    constexpr static char const *name = "matmul_l1";
    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
        matmul_l1::launch_matmul_l1(size_i, size_j, size_k, a, b, c);
    }
};

struct MatmulL1Reg {
    constexpr static char const *name = "matmul_l1_reg";
    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
        matmul_l1_reg::launch_matmul_l1_reg(size_i, size_j, size_k, a, b, c);
    }
};

int main(int argc, char **argv) {
    std::string test_data_dir = ".";
    if (char *c_str_test_data_dir = std::getenv("MATMUL_TEST_DATA_DIR")) {
        test_data_dir = c_str_test_data_dir;
    }

    auto saved_results = std::vector<BenchmarkResult>();

    run_all_tests<MatmulL1>(test_data_dir, saved_results);
    run_all_tests<MatmulL1Reg>(test_data_dir, saved_results);

    if (saved_results.size() > 1) {
        printf("speedups on largest problem size:\n");
        for (int32_t j = 1; j < saved_results.size(); ++j) {
            printf("\n");
            for (int32_t i = j; i > 0;) {
                --i;
                auto const &first = saved_results.at(i);
                auto const &second = saved_results.at(j);
                printf(
                    "  speedup %s -> %s: %.02fx\n",
                    first.name,
                    second.name,
                    first.elapsed_ms / second.elapsed_ms);
            }
        }
    }

    return 0;
}