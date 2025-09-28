// Tested on RTX A4000
// nvcc -O3 -std=c++17 -gencode arch=compute_86,code=sm_86 -o rle_compress rle_compress.cu
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// Utility Functions

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
// CPU Reference Implementation (Already Written)

void rle_compress_cpu(
    uint32_t raw_count,
    char const *raw,
    std::vector<char> &compressed_data,
    std::vector<uint32_t> &compressed_lengths) {
    compressed_data.clear();
    compressed_lengths.clear();

    uint32_t i = 0;
    while (i < raw_count) {
        char c = raw[i];
        uint32_t run_length = 1;
        i++;
        while (i < raw_count && raw[i] == c) {
            run_length++;
            i++;
        }
        compressed_data.push_back(c);
        compressed_lengths.push_back(run_length);
    }
}

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

namespace rle_gpu {

// utility
__device__ __forceinline__ int ceil_log2(uint32_t x) {
    return 32 - __clz(x - 1);
}

__device__ __forceinline__ int pow2(int i) {
    return 1 << i;
}

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

struct SumOp {
    using Data = uint32_t;

    static __host__ __device__ __forceinline__ Data identity() { return 0; }

    static __host__ __device__ __forceinline__ Data combine(Data a, Data b) {
        return a + b;
    }

    static std::string to_string(Data d) { return std::to_string(d); }
};

// scan implementation
constexpr int NUM_ELEMENTS_PER_THREAD = 12;

// scan helpers
template <typename Op>
__device__ __forceinline__ void scan_per_thread(typename Op::Data* shmem, int start_index) {
    using Data = typename Op::Data;
    Data accum = Op::identity();

    // initial accumulation
    #pragma unroll
    for (int i = 0; i < NUM_ELEMENTS_PER_THREAD; i++) {
        accum = Op::combine(accum, shmem[start_index + i]);
        shmem[start_index + i] = accum;
    }
    __syncthreads();

    // continuing accumulations
    // int curr_idx = start_index - 1;
    for (int i = 0; i < ceil_log2(blockDim.x); i++) {
        int back_idx = NUM_ELEMENTS_PER_THREAD * (pow2(i) - 1) + 1;
        int curr_idx = start_index - back_idx;

        Data prev_accum = (curr_idx >= 0) ? shmem[curr_idx] : Op::identity();
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < NUM_ELEMENTS_PER_THREAD; j++) {
            shmem[start_index + j] = Op::combine(prev_accum, shmem[start_index + j]);
        }
        __syncthreads();
    }
}

template <typename Op>
__device__ __forceinline__ void compute_boundaries(typename Op::Data* shmem, const char* src, int start_index, int n) {
    using Data = typename Op::Data;
    const int block_size = NUM_ELEMENTS_PER_THREAD * blockDim.x;
    const int j = blockIdx.x * block_size + start_index;
    Data prev_byte;
    bool first = false;
    if (start_index != 0) {
        prev_byte = shmem[start_index - 1];
    } else {
        if (blockIdx.x != 0) {
            prev_byte = src[blockIdx.x * block_size - 1];
        } else {
            first = true;
        }
    }
    __syncthreads();
    Data curr_byte;
    for (int i = 0; (i < NUM_ELEMENTS_PER_THREAD) && (i + j < n); i++) {
        curr_byte = shmem[start_index + i];
        shmem[start_index + i] = first || (curr_byte != prev_byte);

        first = false;
        prev_byte = curr_byte;
    }

    // if (threadIdx.x == 1 && blockIdx.x == 0) {
    //     for (int i = 0; i < NUM_ELEMENTS_PER_THREAD; i++) {
    //         printf("i: %d | shmem: %d\n", i, shmem[start_index + i]);
    //     }
    // }
}

// scan main
template <typename Op>
__global__ void scan_block_compute_boundaries(size_t n, const char* src, typename Op::Data* dest, typename Op::Data* endpoints) {
    using Data = typename Op::Data;
    extern __shared__ __align__(16) char shmem_raw[];
    Data* shmem = reinterpret_cast<Data*>(shmem_raw);

    const int block_size = NUM_ELEMENTS_PER_THREAD * blockDim.x;
    const int j = blockIdx.x * block_size + threadIdx.x;

    // load into shared memory
    #pragma unroll
    for (int shift_idx = 0; shift_idx < block_size; shift_idx += blockDim.x) {
        shmem[shift_idx + threadIdx.x] = (shift_idx + j < n) ? src[shift_idx + j] : Op::identity();
    }
    __syncthreads();

    const int start_index = threadIdx.x * NUM_ELEMENTS_PER_THREAD;

    compute_boundaries<Op>(shmem, src, start_index, n);

    // scan
    scan_per_thread<Op>(shmem, start_index);

    // store block
    #pragma unroll
    for (int shift_idx = 0; shift_idx < block_size; shift_idx += blockDim.x) {
        dest[blockIdx.x * block_size + shift_idx + threadIdx.x] = shmem[shift_idx + threadIdx.x];
    }

    if (gridDim.x == 1) return;  // if we have only one block, no need to perform a hierarchical scan
    
    // store endpoints
    if (threadIdx.x == blockDim.x - 1) {
        endpoints[blockIdx.x] = shmem[start_index + NUM_ELEMENTS_PER_THREAD - 1];
    }
}

template <typename Op>
__global__ void scan_block(size_t n, typename Op::Data* src, typename Op::Data* dest, typename Op::Data* endpoints) {
    using Data = typename Op::Data;
    extern __shared__ __align__(16) char shmem_raw[];
    Data* shmem = reinterpret_cast<Data*>(shmem_raw);

    const int block_size = NUM_ELEMENTS_PER_THREAD * blockDim.x;
    const int i = blockIdx.x * block_size + threadIdx.x;

    // load into shared memory
    #pragma unroll
    for (int shift_idx = 0; shift_idx < block_size; shift_idx += blockDim.x) {
        shmem[shift_idx + threadIdx.x] = (shift_idx + i < n) ? src[shift_idx + i] : Op::identity();
    }
    __syncthreads();
    // scan
    const int start_index = threadIdx.x * NUM_ELEMENTS_PER_THREAD;
    scan_per_thread<Op>(shmem, start_index);

    // store block
    #pragma unroll
    for (int shift_idx = 0; shift_idx < block_size; shift_idx += blockDim.x) {
        dest[blockIdx.x * block_size + shift_idx + threadIdx.x] = shmem[shift_idx + threadIdx.x];
    }

    if (gridDim.x == 1) return;  // if we have only one block, no need to perform a hierarchical scan
    
    // store endpoints
    if (threadIdx.x == blockDim.x - 1) {
        endpoints[blockIdx.x] = shmem[start_index + NUM_ELEMENTS_PER_THREAD - 1];
    }
}

__device__ __forceinline__ void rle_compress(
    uint32_t* output_idxs, 
    const char* raw, 
    char* compressed_data, 
    uint32_t* compressed_lengths,
    int j,
    int n) {

    char curr_data = raw[0];
    uint32_t run_length = 1;
    uint32_t output_idx;

    if (j >= n) return;

    int i = 1;
    for (; (i < NUM_ELEMENTS_PER_THREAD) && (i + j < n); i++) {
        char next_data = raw[i];

        if (curr_data != next_data) {
            output_idx = output_idxs[i-1] - 1;
            compressed_data[output_idx] = curr_data;
            atomicAdd(&compressed_lengths[output_idx], run_length);
            run_length = 0;
        }

        curr_data = next_data;
        run_length += 1;
    }

    output_idx = output_idxs[i - 1] - 1;
    compressed_data[output_idx] = curr_data;
    atomicAdd(&compressed_lengths[output_idx], run_length);
}

template <typename Op>
__global__ void fixup_rle_compress(
    size_t n, 
    typename Op::Data* workspace, 
    typename Op::Data* endpoints, 
    const char* raw_data, 
    char* compressed_data,
    uint32_t* compressed_lengths) {

    using Data = typename Op::Data;

    const int block_size = blockDim.x * NUM_ELEMENTS_PER_THREAD;
    const int j = blockIdx.x * block_size + threadIdx.x * NUM_ELEMENTS_PER_THREAD;
    const Data local_scan_val = (blockIdx.x != 0) ? endpoints[blockIdx.x - 1] : Op::identity();

    Data output_idxs[NUM_ELEMENTS_PER_THREAD];
    char raw[NUM_ELEMENTS_PER_THREAD];

    // compute output_idxs via fix up and store them into registers
    #pragma unroll
    for (int i = 0; i < NUM_ELEMENTS_PER_THREAD; i++) {
        output_idxs[i] = Op::combine(local_scan_val, workspace[j+i]);
    }

    // load raw_data into registers
    #pragma unroll
    for (int i = 0; i < NUM_ELEMENTS_PER_THREAD; i++) {
        if (j + i < n) {
            raw[i] = raw_data[j + i];
        }
    }

    rle_compress(output_idxs, raw, compressed_data, compressed_lengths, j, n);

    // store raw count
    if ((blockIdx.x == gridDim.x - 1) && (threadIdx.x == blockDim.x - 1)) {
        endpoints[blockIdx.x] = output_idxs[NUM_ELEMENTS_PER_THREAD - 1];
    }
}


// Returns desired size of scratch buffer in bytes.
size_t get_workspace_size(uint32_t raw_count) {
    constexpr int thread_per_block = 1024;
    const int block_size = NUM_ELEMENTS_PER_THREAD * thread_per_block;
    const int num_blocks = (raw_count + (block_size - 1)) / block_size;
    return sizeof(uint32_t) * ((num_blocks + 1) * block_size);
}

// 'launch_rle_compress'
//
// Input:
//
//   'raw_count': Number of bytes in the input buffer 'raw'.
//
//   'raw': Uncompressed bytes in GPU memory.
//
//   'workspace': Scratch buffer in GPU memory. The size of the scratch buffer
//   in bytes is determined by 'get_workspace_size'.
//
// Output:
//
//   Returns: 'compressed_count', the number of runs in the compressed data.
//
//   'compressed_data': Output buffer of size 'raw_count' in GPU memory. The
//   function should fill the first 'compressed_count' bytes of this buffer
//   with the compressed data.
//
//   'compressed_lengths': Output buffer of size 'raw_count' in GPU memory. The
//   function should fill the first 'compressed_count' integers in this buffer
//   with the lengths of the runs in the compressed data.
//
uint32_t launch_rle_compress(
    uint32_t raw_count,
    char const *raw,             // pointer to GPU buffer
    void *workspace,             // pointer to GPU buffer
    char *compressed_data,       // pointer to GPU buffer
    uint32_t *compressed_lengths // pointer to GPU buffer
) {
    dim3 block_size_threads(1024);
    const int block_size = NUM_ELEMENTS_PER_THREAD * block_size_threads.x;
    dim3 num_blocks((raw_count + block_size - 1) / block_size);

    uint32_t shmem_bytes = block_size * sizeof(uint32_t);

    CUDA_CHECK(cudaFuncSetAttribute(
        scan_block_compute_boundaries<SumOp>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        shmem_bytes));
    
    CUDA_CHECK(cudaFuncSetAttribute(
        scan_block<SumOp>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        shmem_bytes));

    uint32_t* workspace_data = (uint32_t*) workspace;
    uint32_t* endpoints = workspace_data + num_blocks.x * block_size;

    scan_block_compute_boundaries<SumOp><<<num_blocks, block_size_threads, shmem_bytes>>>((size_t) raw_count, raw, workspace_data, endpoints);
    CUDA_CHECK(cudaGetLastError());

    scan_block<SumOp><<<1, block_size_threads, shmem_bytes>>>((size_t) num_blocks.x, endpoints, endpoints, (uint32_t*) NULL);
    CUDA_CHECK(cudaGetLastError());

    fixup_rle_compress<SumOp><<<num_blocks, block_size_threads>>>(
        (size_t) raw_count, 
        workspace_data, 
        endpoints, 
        raw, 
        compressed_data, 
        compressed_lengths
    );

    CUDA_CHECK(cudaGetLastError());

    uint32_t compressed_count;

    CUDA_CHECK(cudaMemcpy(
        &compressed_count,
        &endpoints[num_blocks.x - 1],
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost
    ));

    return compressed_count;
}

} // namespace rle_gpu

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

template <typename Reset, typename F>
double benchmark_ms(double target_time_ms, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        f();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms);
    }
    return best_time_ms;
}

struct Results {
    double time_ms;
};

enum class Mode {
    TEST,
    BENCHMARK,
};

Results run_config(Mode mode, std::vector<char> const &raw) {
    // Allocate buffers
    size_t workspace_size = rle_gpu::get_workspace_size(raw.size());
    char *raw_gpu;
    void *workspace;
    char *compressed_data_gpu;
    uint32_t *compressed_lengths_gpu;
    CUDA_CHECK(cudaMalloc(&raw_gpu, raw.size()));
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    CUDA_CHECK(cudaMalloc(&compressed_data_gpu, raw.size()));
    CUDA_CHECK(cudaMalloc(&compressed_lengths_gpu, raw.size() * sizeof(uint32_t)));

    // Copy input data to GPU
    CUDA_CHECK(cudaMemcpy(raw_gpu, raw.data(), raw.size(), cudaMemcpyHostToDevice));

    auto reset = [&]() {
        CUDA_CHECK(cudaMemset(compressed_data_gpu, 0, raw.size()));
        CUDA_CHECK(cudaMemset(compressed_lengths_gpu, 0, raw.size() * sizeof(uint32_t)));
    };

    auto f = [&]() {
        rle_gpu::launch_rle_compress(
            raw.size(),
            raw_gpu,
            workspace,
            compressed_data_gpu,
            compressed_lengths_gpu);
    };

    // Test correctness
    reset();
    uint32_t compressed_count = rle_gpu::launch_rle_compress(
        raw.size(),
        raw_gpu,
        workspace,
        compressed_data_gpu,
        compressed_lengths_gpu);
    std::vector<char> compressed_data(compressed_count);
    std::vector<uint32_t> compressed_lengths(compressed_count);
    CUDA_CHECK(cudaMemcpy(
        compressed_data.data(),
        compressed_data_gpu,
        compressed_count,
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        compressed_lengths.data(),
        compressed_lengths_gpu,
        compressed_count * sizeof(uint32_t),
        cudaMemcpyDeviceToHost));

    std::vector<char> compressed_data_expected;
    std::vector<uint32_t> compressed_lengths_expected;
    rle_compress_cpu(
        raw.size(),
        raw.data(),
        compressed_data_expected,
        compressed_lengths_expected);

    bool correct = true;
    if (compressed_count != compressed_data_expected.size()) {
        printf("Mismatch in compressed count:\n");
        printf("  Expected: %zu\n", compressed_data_expected.size());
        printf("  Actual:   %u\n", compressed_count);
        correct = false;
    }
    if (correct) {
        for (size_t i = 0; i < compressed_data_expected.size(); i++) {
            if (compressed_data[i] != compressed_data_expected[i]) {
                printf("Mismatch in compressed data at index %zu:\n", i);
                printf(
                    "  Expected: 0x%02x\n",
                    static_cast<unsigned char>(compressed_data_expected[i]));
                printf(
                    "  Actual:   0x%02x\n",
                    static_cast<unsigned char>(compressed_data[i]));
                correct = false;
                break;
            }
            if (compressed_lengths[i] != compressed_lengths_expected[i]) {
                printf("Mismatch in compressed lengths at index %zu:\n", i);
                printf("  Expected: %u\n", compressed_lengths_expected[i]);
                printf("  Actual:   %u\n", compressed_lengths[i]);
                correct = false;
                break;
            }
        }
    }
    if (!correct) {
        if (raw.size() <= 1024) {
            printf("\nInput:\n");
            for (size_t i = 0; i < raw.size(); i++) {
                printf("  [%4zu] = 0x%02x\n", i, static_cast<unsigned char>(raw[i]));
            }
            printf("\nExpected:\n");
            for (size_t i = 0; i < compressed_data_expected.size(); i++) {
                printf(
                    "  [%4zu] = data: 0x%02x, length: %u\n",
                    i,
                    static_cast<unsigned char>(compressed_data_expected[i]),
                    compressed_lengths_expected[i]);
            }
            printf("\nActual:\n");
            if (compressed_data.size() == 0) {
                printf("  (empty)\n");
            }
            for (size_t i = 0; i < compressed_data.size(); i++) {
                printf(
                    "  [%4zu] = data: 0x%02x, length: %u\n",
                    i,
                    static_cast<unsigned char>(compressed_data[i]),
                    compressed_lengths[i]);
            }
        }
        exit(1);
    }

    if (mode == Mode::TEST) {
        return {};
    }

    // Benchmark
    double target_time_ms = 1000.0;
    double time_ms = benchmark_ms(target_time_ms, reset, f);

    // Cleanup
    CUDA_CHECK(cudaFree(raw_gpu));
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(compressed_data_gpu));
    CUDA_CHECK(cudaFree(compressed_lengths_gpu));

    return {time_ms};
}

template <typename Rng> std::vector<char> generate_test_data(uint32_t size, Rng &rng) {
    auto random_byte = std::uniform_int_distribution<int32_t>(
        std::numeric_limits<char>::min(),
        std::numeric_limits<char>::max());
    constexpr uint32_t alphabet_size = 4;
    auto alphabet = std::vector<char>();
    for (uint32_t i = 0; i < alphabet_size; i++) {
        alphabet.push_back(random_byte(rng));
    }
    auto random_symbol = std::uniform_int_distribution<uint32_t>(0, alphabet_size - 1);
    auto data = std::vector<char>();
    for (uint32_t i = 0; i < size; i++) {
        data.push_back(alphabet.at(random_symbol(rng)));
    }
    return data;
}

int main(int argc, char const *const *argv) {
    auto rng = std::mt19937(0xCA7CAFE);

    auto test_sizes = std::vector<uint32_t>{
        16,
        10,
        128,
        100,
        1 << 10,
        1000,
        1 << 20,
        1'000'000,
        16 << 20,
    };

    printf("Correctness:\n\n");
    for (auto test_size : test_sizes) {
        auto raw = generate_test_data(test_size, rng);
        printf("  Testing compression for size %u\n", test_size);
        run_config(Mode::TEST, raw);
        printf("  OK\n\n");
    }

    auto test_data_search_paths = std::vector<std::string>{".", "/"};
    std::string test_data_path;
    for (auto test_data_search_path : test_data_search_paths) {
        auto candidate_path = test_data_search_path + "/rle_raw.bmp";
        if (std::filesystem::exists(candidate_path)) {
            test_data_path = candidate_path;
            break;
        }
    }
    if (test_data_path.empty()) {
        printf("Could not find test data file.\n");
        exit(1);
    }

    auto raw = std::vector<char>();
    {
        auto file = std::ifstream(test_data_path, std::ios::binary);
        if (!file) {
            printf("Could not open test data file '%s'.\n", test_data_path.c_str());
            exit(1);
        }
        file.seekg(0, std::ios::end);
        raw.resize(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(raw.data(), raw.size());
    }

    printf("Performance:\n\n");
    printf("  Testing compression on file 'rle_raw.bmp' (size %zu)\n", raw.size());
    auto results = run_config(Mode::BENCHMARK, raw);
    printf("  Time: %.2f ms\n", results.time_ms);

    return 0;
}