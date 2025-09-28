// Tested on RTX A4000
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

__device__ inline void cp_async4(const void *smem_ptr, const void *glob_ptr) {
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
    float* b_shared = &shmem[2 * tileDim.x * tileDim.y];
    
    if (i >= size_i || j >= size_j) return;

    float a_microtile[K] = {0};
    float b_microtile[K] = {0};
    
    float sum[K][K] = {0};

    uint32_t tile_iters = (size_k + tileDim.x - 1) / tileDim.x;

    for (int tile_idx = 0; tile_idx < tile_iters; tile_idx += 2) {
        // load into shared memory
        #pragma unroll
        for (int y = 0; y < K; y++) {
            #pragma unroll
            for (int x = 0; x < K; x++) {
                a_shared[(t.y+y)*(2*tileDim.x) + (t.x+x)] = a[(i+y)*size_k + (tile_idx*tileDim.x + t.x+x)];
                a_shared[(t.y+y)*(2*tileDim.x) + (t.x+x + tileDim.x)] = a[(i+y)*size_k + ((tile_idx+1)*tileDim.x + t.x+x)];

                b_shared[(t.y+y)*tileDim.x + (t.x+x)] = b[(tile_idx * tileDim.y + t.y+y)*size_j + (j+x)];
                b_shared[(t.y+y + tileDim.y)*tileDim.x + (t.x+x)] = b[((tile_idx+1) * tileDim.y + t.y+y)*size_j + (j+x)];
            }
        }
        __syncthreads();
        for (int microtile_idx = 0; microtile_idx < tileDim.x*2; microtile_idx++) {
            // load into microtile
            #pragma unroll
            for (int y = 0; y < K; y++) {
                a_microtile[y] = a_shared[(t.y+y)*(2*tileDim.x) + (microtile_idx)];
                b_microtile[y] = b_shared[(microtile_idx)*tileDim.x + (t.x+y)];
            }

            // compute microtile
            #pragma unroll
            for (int y = 0; y < K; y++) {
                #pragma unroll
                for (int x = 0; x < K; x++) {
                    sum[y][x] += a_microtile[y] * b_microtile[x];
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
    
    uint32_t shmem_bytes = 2 * 2 * tile_size.x * tile_size.y * sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute(
        matmul_l1_reg,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        shmem_bytes));
    
    matmul_l1_reg<<<num_blocks, block_size, shmem_bytes>>>(size_i, size_j, size_k, a, b, c);
}

} // namespace matmul_l1_reg

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

__device__ __forceinline__ void swap(float* &a, float* &b) {
    float* temp = a;
    a = b;
    b = temp;
}

namespace matmul_improved {

constexpr int MICROTILE_SIZE = 4;
constexpr int REDUCE_DIM = 32;
constexpr int MICROTILE_REDUCE_DIM = 4;

__device__ __forceinline__ void load_tiles_to_shared_fp4(
    float* a_shared, 
    float* b_shared, 
    const float* a, 
    const float* b, 
    int size_i,
    int size_j,
    int size_k,
    int tile_idx,
    dim3 tileDim) {

    const int threadIdx_lin = 4 * ((threadIdx.y * blockDim.x) + threadIdx.x); // linearize

    const int y_a = tileDim.y*blockIdx.y + (threadIdx_lin / REDUCE_DIM);
    const int x_a = tile_idx*REDUCE_DIM + (threadIdx_lin % REDUCE_DIM);

    const int y_b = tile_idx*REDUCE_DIM + (threadIdx_lin / tileDim.x);
    const int x_b = tileDim.x*blockIdx.x + (threadIdx_lin % tileDim.x);

    void* a_shmem_addr = a_shared + threadIdx_lin;
    void* b_shmem_addr = b_shared + threadIdx_lin;

    if (y_a < size_i && x_a < size_k) {
        const void* a_gmem_addr = a + ((y_a)*size_k + (x_a));
        cp_async4(a_shmem_addr, a_gmem_addr);
    } else {
        reinterpret_cast<float4*>(a_shmem_addr)[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    if (y_b < size_k && x_b < size_j) {
        const void* b_gmem_addr = b + ((y_b)*size_j + (x_b));
        cp_async4(b_shmem_addr, b_gmem_addr);
    } else {
        reinterpret_cast<float4*>(b_shmem_addr)[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

}


__device__ __forceinline__ void reduce(
    uint2 t, dim3 tileDim, const float* a_shared, const float* b_shared, float sum[MICROTILE_SIZE][MICROTILE_SIZE]
) {
    float a_microtile[MICROTILE_SIZE][MICROTILE_REDUCE_DIM] = {0};
    float b_microtile[MICROTILE_REDUCE_DIM][MICROTILE_SIZE] = {0};

    for (int microtile_idx = 0; microtile_idx < REDUCE_DIM; microtile_idx += MICROTILE_REDUCE_DIM) {
        // load into microtile
        #pragma unroll
        for (int i = 0; i < MICROTILE_SIZE; i++) {
            #pragma unroll
            for (int k = 0; k < MICROTILE_REDUCE_DIM; k++) {
                a_microtile[i][k] = a_shared[(t.y+i)*REDUCE_DIM + (microtile_idx + k)];
                b_microtile[k][i] = b_shared[(microtile_idx + k)*tileDim.x + (t.x+i)];
            }
        }

        // compute microtile
        #pragma unroll
        for (int i = 0; i < MICROTILE_SIZE; i++) {
            #pragma unroll
            for (int j = 0; j < MICROTILE_SIZE; j++) {
                #pragma unroll
                for (int k = 0; k < MICROTILE_REDUCE_DIM; k++) {
                    sum[i][j] += a_microtile[i][k] * b_microtile[k][j];
                }
            }
        }
    }
}

__global__ void matmul_improved(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *c /* pointer to GPU memory */) {

    const dim3 tileDim{MICROTILE_SIZE * blockDim.x, MICROTILE_SIZE * blockDim.y};

    const uint2 t{MICROTILE_SIZE * threadIdx.x, MICROTILE_SIZE * threadIdx.y};
    const uint32_t y = tileDim.y * blockIdx.y + t.y;
    const uint32_t x = tileDim.x * blockIdx.x + t.x;

    extern __shared__ float shmem[];

    const int tile_size = tileDim.x * REDUCE_DIM;
    float* a_shared0 = shmem;              // (tileDim.y x REDUCE_DIM)
    float* b_shared0 = &shmem[tile_size];  // (REDUCE_DIM x tileDim.x)

    float* a_shared1 = &shmem[2 * tile_size];
    float* b_shared1 = &shmem[3 * tile_size];
    
    // if (i >= size_i || j >= size_j) return;

    float sum[MICROTILE_SIZE][MICROTILE_SIZE] = {0};

    uint32_t tile_iters = (size_k + REDUCE_DIM - 1) / REDUCE_DIM;

    // load inital tiles
    load_tiles_to_shared_fp4(a_shared0, b_shared0, a, b, size_i, size_j, size_k, 0, tileDim);
    async_memcpy_waitall();
    __syncthreads();

    for (int tile_idx = 1; tile_idx < tile_iters; tile_idx++) {
        // load in next tile
        load_tiles_to_shared_fp4(a_shared1, b_shared1, a, b, size_i, size_j, size_k, tile_idx, tileDim);

        reduce(t, tileDim, a_shared0, b_shared0, sum);

        swap(a_shared0, a_shared1);
        swap(b_shared0, b_shared1);

        async_memcpy_waitall();
        __syncthreads();
    }

    // last last tile
    reduce(t, tileDim, a_shared0, b_shared0, sum);
    
    // store microtile sums
    int y_idx;
    int x_idx;
    #pragma unroll
    for (int i = 0; i < MICROTILE_SIZE; i++) {
        #pragma unroll
        for (int j = 0; j < MICROTILE_SIZE; j++) {
            y_idx = y + i;
            x_idx = x + j;
            if (((y_idx) < size_i) && ((x_idx) < size_j)) {
                c[(y_idx)*size_j + (x_idx)] = sum[i][j];
            }
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
    dim3 tile_size(MICROTILE_SIZE * block_size.x, MICROTILE_SIZE * block_size.y); // does block_size * k work?
    dim3 num_blocks((size_j + tile_size.x - 1) / tile_size.x,
                    (size_i + tile_size.y - 1) / tile_size.y);
    
    uint32_t shmem_bytes = 2 * 2 * tile_size.x * REDUCE_DIM * sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute(
        matmul_improved,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        shmem_bytes));
    
    matmul_improved<<<num_blocks, block_size, shmem_bytes>>>(size_i, size_j, size_k, a, b, c);

    CUDA_CHECK(cudaGetLastError());
}

}; // namespace matmul_improved

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation with Reduction along k

namespace matmul_improved_reduce {

constexpr int MICROTILE_SIZE = 4;
constexpr int REDUCE_DIM = 32;
constexpr int K_SPLIT_SIZE = 768;  // multiple of REDUCE_SIZE

/* TODO: your GPU kernels here... */
__global__ void matmul_improved_reducek(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *workspace /* pointer to GPU memory */) {

    const dim3 tileDim{MICROTILE_SIZE * blockDim.x, MICROTILE_SIZE * blockDim.y};
    float* c = workspace + blockIdx.z*size_i*size_j;

    const uint2 t{MICROTILE_SIZE * threadIdx.x, MICROTILE_SIZE * threadIdx.y};
    const uint32_t y = tileDim.y * blockIdx.y + t.y;
    // const uint32_t y = tileDim.y * blockIdx.y + threadIdx.y;
    const uint32_t x = tileDim.x * blockIdx.x + t.x;
    // const uint32_t x = tileDim.x * blockIdx.x + threadIdx.x;

    extern __shared__ float shmem[];

    const int tile_size = tileDim.x * REDUCE_DIM;
    float* a_shared0 = shmem;              // (tileDim.y x REDUCE_DIM)
    float* b_shared0 = &shmem[tile_size];  // (REDUCE_DIM x tileDim.x)

    float* a_shared1 = &shmem[2 * tile_size];
    float* b_shared1 = &shmem[3 * tile_size];
    
    // if (i >= size_i || j >= size_j) return;

    float sum[MICROTILE_SIZE][MICROTILE_SIZE] = {0};

    const uint32_t tile_iters = K_SPLIT_SIZE / REDUCE_DIM;
    const int start_tile_iter = (blockIdx.z) * K_SPLIT_SIZE / REDUCE_DIM;

    // load inital tiles
    matmul_improved::load_tiles_to_shared_fp4(a_shared0, b_shared0, a, b, size_i, size_j, size_k, start_tile_iter, tileDim);
    async_memcpy_waitall();
    __syncthreads();

    for (int tile_idx = 1; tile_idx < tile_iters; tile_idx++) {
        // load in next tile
        matmul_improved::load_tiles_to_shared_fp4(a_shared1, b_shared1, a, b, size_i, size_j, size_k, start_tile_iter + tile_idx, tileDim);

        matmul_improved::reduce(t, tileDim, a_shared0, b_shared0, sum);

        swap(a_shared0, a_shared1);
        swap(b_shared0, b_shared1);

        async_memcpy_waitall();
        __syncthreads();
    }

    // last last tile
    matmul_improved::reduce(t, tileDim, a_shared0, b_shared0, sum);

    // store microtile sums
    #pragma unroll
    for (int i = 0; i < MICROTILE_SIZE; i++) {
        #pragma unroll
        for (int j = 0; j < MICROTILE_SIZE; j++) {
            const int y_idx = y + i;
            const int x_idx = x + j;
            if (y_idx < size_i && x_idx < size_j) {
                c[(y_idx)*size_j + (x_idx)] = sum[i][j];
            }
        }
    }
}

__global__ void reduce_k(int32_t size_i, int32_t size_j, float* workspace, float* c) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < size_i && x < size_j) {
        float sum = 0;
        for (int i = 0; i < gridDim.z; i++) {
            sum += (workspace + i*size_i*size_j)[(y)*size_j + (x)];
        }
        c[(y)*size_j + (x)] = sum;
    }
}

size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
    /* TODO: your CPU code here */
    const int num_splits = (size_k + K_SPLIT_SIZE - 1) / K_SPLIT_SIZE;
    return size_i * size_j * num_splits * 4;
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
    const int num_splits = (size_k + K_SPLIT_SIZE - 1) / K_SPLIT_SIZE;
    // std::cerr << "num_splits: " << num_splits << '\n';
    dim3 block_size(32, 32);
    dim3 tile_size(MICROTILE_SIZE * block_size.x, MICROTILE_SIZE * block_size.y); // does block_size * k work?
    dim3 num_blocks_matmul(
        (size_j + tile_size.x - 1) / tile_size.x,
        (size_i + tile_size.y - 1) / tile_size.y,
        num_splits
    );
    
    uint32_t shmem_bytes = 2 * 2 * tile_size.x * REDUCE_DIM * sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute(
        matmul_improved_reducek,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        shmem_bytes));
    
    matmul_improved_reducek<<<num_blocks_matmul, block_size, shmem_bytes>>>(size_i, size_j, size_k, a, b, (float*) workspace);
    CUDA_CHECK(cudaGetLastError());


    dim3 num_blocks_reduce_k(
        (size_j + block_size.x - 1) / block_size.x,
        (size_i + block_size.y - 1) / block_size.y,
        num_splits
    );

    reduce_k<<<num_blocks_reduce_k, block_size>>>(size_i, size_j, (float*) workspace, c);
    CUDA_CHECK(cudaGetLastError());
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
        {512, 3072, 3072},
        {256, 3072, 3072},
        {128, 3072, 3072},
        {64, 3072, 3072},
        {32, 3072, 3072},
        {16, 3072, 3072},
        {1, 3072, 3072},
        {256, 256, 256},
        {256, 256, 1024},
        {256, 256, 8192},
        {128, 128, 32768},
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