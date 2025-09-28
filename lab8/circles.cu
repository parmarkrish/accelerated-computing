// Tested on RTX A4000
// nvcc -O3 -std=c++17 -gencode arch=compute_86,code=sm_86 -o circles circles.cu
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>
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

class GpuMemoryPool {
  public:
    GpuMemoryPool() = default;

    ~GpuMemoryPool();

    GpuMemoryPool(GpuMemoryPool const &) = delete;
    GpuMemoryPool &operator=(GpuMemoryPool const &) = delete;
    GpuMemoryPool(GpuMemoryPool &&) = delete;
    GpuMemoryPool &operator=(GpuMemoryPool &&) = delete;

    void *alloc(size_t size);
    void reset();

  private:
    std::vector<void *> allocations_;
    std::vector<size_t> capacities_;
    size_t next_idx_ = 0;
};

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Already Written)

void render_cpu(
    int32_t width,
    int32_t height,
    int32_t n_circle,
    float const *circle_x,
    float const *circle_y,
    float const *circle_radius,
    float const *circle_red,
    float const *circle_green,
    float const *circle_blue,
    float const *circle_alpha,
    float *img_red,
    float *img_green,
    float *img_blue) {

    // Initialize background to white
    for (int32_t pixel_idx = 0; pixel_idx < width * height; pixel_idx++) {
        img_red[pixel_idx] = 1.0f;
        img_green[pixel_idx] = 1.0f;
        img_blue[pixel_idx] = 1.0f;
    }

    // Render circles
    for (int32_t i = 0; i < n_circle; i++) {
        float c_x = circle_x[i];
        float c_y = circle_y[i];
        float c_radius = circle_radius[i];
        for (int32_t y = int32_t(c_y - c_radius); y <= int32_t(c_y + c_radius + 1.0f);
             y++) {
            for (int32_t x = int32_t(c_x - c_radius); x <= int32_t(c_x + c_radius + 1.0f);
                 x++) {
                float dx = x - c_x;
                float dy = y - c_y;
                if (!(0 <= x && x < width && 0 <= y && y < height &&
                      dx * dx + dy * dy < c_radius * c_radius)) {
                    continue;
                }
                int32_t pixel_idx = y * width + x;
                float pixel_red = img_red[pixel_idx];
                float pixel_green = img_green[pixel_idx];
                float pixel_blue = img_blue[pixel_idx];
                float pixel_alpha = circle_alpha[i];
                pixel_red =
                    circle_red[i] * pixel_alpha + pixel_red * (1.0f - pixel_alpha);
                pixel_green =
                    circle_green[i] * pixel_alpha + pixel_green * (1.0f - pixel_alpha);
                pixel_blue =
                    circle_blue[i] * pixel_alpha + pixel_blue * (1.0f - pixel_alpha);
                img_red[pixel_idx] = pixel_red;
                img_green[pixel_idx] = pixel_green;
                img_blue[pixel_idx] = pixel_blue;
            }
        }
    }
}

/// <--- your code here --->

// utils

int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ int ceil_log2(uint32_t x) {
    return 32 - __clz(x - 1);
}

__device__ __forceinline__ int pow2(int i) {
    return 1 << i;
}

struct SumOp {
    using Data = uint32_t;

    static __host__ __device__ __forceinline__ Data identity() { return 0; }

    static __host__ __device__ __forceinline__ Data combine(Data a, Data b) {
        return a + b;
    }

    static std::string to_string(Data d) { return std::to_string(d); }
};


#define KERNEL_PRINTF(...) \
  do { \
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && \
        threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) { \
      printf(__VA_ARGS__); \
    } \
  } while (0)

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

struct CircleInfo {
    float cx;
    float cy;
    float c_radius;
    float c_r;
    float c_g;
    float c_b;
    float pixel_alpha;
};

namespace circles_gpu {
    __device__ __forceinline__ bool intersect(
        int x_min, int x_max, int y_min, int y_max,
        float circles_x, float circles_y, float radius
    );
}

namespace scan
{

constexpr int NUM_ELEMENTS_PER_THREAD = 12;

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
__global__ void scan_block_circle_intersect(
    size_t n, 
    const float* circle_x, 
    const float* circle_y, 
    const float* radius,
    dim3 draw_blockDim,
    dim3 draw_blockIdx,
    typename Op::Data* dest, 
    typename Op::Data* endpoints) {

    using Data = typename Op::Data;
    extern __shared__ __align__(16) char shmem_raw[];
    Data* shmem = reinterpret_cast<Data*>(shmem_raw);

    const int block_size = NUM_ELEMENTS_PER_THREAD * blockDim.x;
    const int l_idx = threadIdx.x * NUM_ELEMENTS_PER_THREAD;
    const int j = blockIdx.x * block_size + l_idx;

    for (int i = 0; i < NUM_ELEMENTS_PER_THREAD; i++) {
        int g_idx = i + j;

        if (g_idx >= n) {
            shmem[l_idx + i] = Op::identity();
            continue;
        }

        float cx = circle_x[g_idx];
        float cy = circle_y[g_idx];
        float r = radius[g_idx];

        int x_min = draw_blockIdx.x * draw_blockDim.x;
        int x_max = (1 + draw_blockIdx.x) * draw_blockDim.x;

        int y_min = draw_blockIdx.y * draw_blockDim.y;
        int y_max = (1 + draw_blockIdx.y) * draw_blockDim.y;

        bool is_intersect = circles_gpu::intersect(x_min, x_max, y_min, y_max, cx, cy, r);

        shmem[l_idx + i] = is_intersect;

        // KERNEL_PRINTF("cx: %f, cy: %f, r: %f\n", cx, cy, r);
        // KERNEL_PRINTF("is_intersect: %d\n", is_intersect);
    }
    __syncthreads();

    scan_per_thread<Op>(shmem, l_idx);

    // store block
    #pragma unroll
    for (int shift_idx = 0; shift_idx < block_size; shift_idx += blockDim.x) {
        dest[blockIdx.x * block_size + shift_idx + threadIdx.x] = shmem[shift_idx + threadIdx.x];
    }

    if (gridDim.x == 1) return;  // if we have only one block, no need to perform a hierarchical scan
    
    // store endpoints
    if (threadIdx.x == blockDim.x - 1) {
        endpoints[blockIdx.x] = shmem[l_idx + NUM_ELEMENTS_PER_THREAD - 1];
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

// avoid loading and storing indicator by detecting change in scan
__device__ __forceinline__ void store_on_change(
    const uint32_t* output_idxs, 
    const float* circle_x,
    const float* circle_y,
    const float* circle_radius,
    const float* circle_red,
    const float* circle_green,
    const float* circle_blue,
    const float* circle_alpha,
    CircleInfo* dest, 
    uint32_t prev_val_,
    int j,
    int n) {
    
    #pragma unroll
    for (int i = 0; (i < NUM_ELEMENTS_PER_THREAD) && (i + j < n); i++) {
        uint32_t prev_val = (i - 1 >= 0) ? output_idxs[i-1] : prev_val_;
        uint32_t curr_val = output_idxs[i];
        if (curr_val != prev_val) {
            int circle_idx = i + j;
            dest[curr_val - 1] = {
                .cx = circle_x[circle_idx],
                .cy = circle_y[circle_idx],
                .c_radius = circle_radius[circle_idx],
                .c_r = circle_red[circle_idx],
                .c_g = circle_green[circle_idx],
                .c_b = circle_blue[circle_idx],
                .pixel_alpha = circle_alpha[circle_idx]
            };
            // dest[curr_val - 1] = i + j;  // store circle index 
            // KERNEL_PRINTF("curr_val: %u, index: %d\n", curr_val, i+j);
        }
    }
}


template <typename Op>
__global__ void fixup_store_circle_idxs(
    size_t n, 
    typename Op::Data* workspace, 
    typename Op::Data* endpoints, 
    const float* circle_x,
    const float* circle_y,
    const float* circle_radius,
    const float* circle_red,
    const float* circle_green,
    const float* circle_blue,
    const float* pixel_alpha,
    CircleInfo* dest,
    uint32_t* draw_block_start_circle_idxs,
    int blockIdx_lin
    ) {
    using Data = typename Op::Data;

    const int block_size = blockDim.x * NUM_ELEMENTS_PER_THREAD;
    const int j = blockIdx.x * block_size + threadIdx.x * NUM_ELEMENTS_PER_THREAD;
    const Data local_scan_val = (blockIdx.x != 0) ? endpoints[blockIdx.x - 1] : Op::identity();

    Data output_idxs[NUM_ELEMENTS_PER_THREAD];

    // compute output_idxs via fix up and store them into registers
    #pragma unroll
    for (int i = 0; i < NUM_ELEMENTS_PER_THREAD; i++) {
        output_idxs[i] = Op::combine(local_scan_val, workspace[j+i]);
        // if (threadIdx.x == 0) {
            // printf("threadIdx.x == %u, blockIdx.x == %u, output_idxs[%d] = %u\n", threadIdx.x, blockIdx.x, i, output_idxs[i]);
        // }
    }

    // store_on_change

    uint32_t prev_val = local_scan_val;
    if (threadIdx.x != 0) {
        prev_val = Op::combine(local_scan_val, workspace[j - 1]);
    }

    uint32_t start_idx = (blockIdx_lin - 1 >= 0) ? draw_block_start_circle_idxs[blockIdx_lin - 1] : 0;
    // KERNEL_PRINTF("start_idx: %u\n", start_idx);
    CircleInfo* dest_start = dest + start_idx;

    store_on_change(
        output_idxs, 
        circle_x,
        circle_y,
        circle_radius,
        circle_red,
        circle_green,
        circle_blue,
        pixel_alpha,
        dest_start, 
        prev_val, 
        j, 
        n
    );

    // store raw count
    if ((blockIdx.x == gridDim.x - 1) && (threadIdx.x == blockDim.x - 1)) {
        uint32_t size = output_idxs[NUM_ELEMENTS_PER_THREAD - 1];
        draw_block_start_circle_idxs[blockIdx_lin] = start_idx + size;
    }
}


} // namespace scan


namespace circles_gpu {

// intersection between circle and square
__device__ __forceinline__ bool intersect(
    int x_min, int x_max, int y_min, int y_max, 
    float circle_x, float circle_y, float radius
) {
    // Project the circle center to the rectangle to find the closest point.
    float qx = fminf(fmaxf(static_cast<float>(x_min), circle_x), static_cast<float>(x_max));
    float qy = fminf(fmaxf(static_cast<float>(y_min), circle_y), static_cast<float>(y_max));

    float dx = circle_x - qx;
    float dy = circle_y - qy;

    return (dx * dx + dy * dy) < radius * radius;
}

// debug
__global__ void print_gpu_mem(
    uint32_t* mem,
    uint32_t start,
    uint32_t end
) {
    for (int i = start; i < end; i++) {
        printf("%d: %d\n", i, mem[i]);
    }
    printf("\n");
}

                // .cx = circle_x[circle_idx],
                // .cy = circle_y[circle_idx],
                // .c_radius = circle_radius[circle_idx],
                // .c_r = circle_red[circle_idx],
                // .c_g = circle_green[circle_idx],
                // .c_b = circle_blue[circle_idx],
                // .pixel_alpha = circle_alpha[circle_idx]
// reorders circle data in a per block in order manner using stream compaction
void reorder_circles_per_block(
    int32_t n_circles, 
    const float* circle_x,
    const float* circle_y,
    const float* circle_radius,
    const float* circle_red,
    const float* circle_green,
    const float* circle_blue,
    const float* pixel_alpha,
    GpuMemoryPool& memory_pool,
    dim3 draw_blockDim,
    dim3 draw_num_blocks,
    CircleInfo* circle_infos,
    uint32_t* draw_block_start_circle_idxs) {

    dim3 blockDim(1024);
    const int block_size = scan::NUM_ELEMENTS_PER_THREAD * blockDim.x;
    const int num_blocks = ceil_div(n_circles, block_size);

    // memory allocation
    size_t num_bytes_workspace = sizeof(uint32_t) * (num_blocks * block_size);
    uint32_t* workspace = reinterpret_cast<uint32_t*>(memory_pool.alloc(num_bytes_workspace));

    size_t num_bytes_endpoints = sizeof(uint32_t) * block_size;
    uint32_t* endpoints = reinterpret_cast<uint32_t*>(memory_pool.alloc(num_bytes_endpoints));



    // setup shared memory
    uint32_t shmem_bytes = block_size * sizeof(uint32_t);
    // std::cerr << shmem_bytes << '\n';

    CUDA_CHECK(cudaFuncSetAttribute(
        scan::scan_block_circle_intersect<SumOp>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        shmem_bytes));
    
    CUDA_CHECK(cudaFuncSetAttribute(
        scan::scan_block<SumOp>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        shmem_bytes));

    for (int y_block_idx = 0; y_block_idx < draw_num_blocks.y; y_block_idx++) {
        for (int x_block_idx = 0; x_block_idx < draw_num_blocks.x; x_block_idx++) {
            int blockIdx_lin = y_block_idx * draw_num_blocks.x + x_block_idx;
            dim3 draw_blockIdx(x_block_idx, y_block_idx);

            // std::cerr << num_blocks << '\n';
            // std::cerr << blockDim.x << '\n';

            scan::scan_block_circle_intersect<SumOp><<<num_blocks, blockDim, shmem_bytes>>>(
                n_circles,
                circle_x,
                circle_y,
                circle_radius,
                draw_blockDim,
                draw_blockIdx,
                workspace,
                endpoints
            );
            CUDA_CHECK(cudaGetLastError());

            scan::scan_block<SumOp><<<1, blockDim, shmem_bytes>>>((size_t) num_blocks, endpoints, endpoints, (uint32_t*) NULL);
            CUDA_CHECK(cudaGetLastError());

            scan::fixup_store_circle_idxs<SumOp><<<num_blocks, blockDim>>>(
                n_circles,
                workspace,
                endpoints,
                circle_x,
                circle_y,
                circle_radius,
                circle_red,
                circle_green,
                circle_blue,
                pixel_alpha,
                circle_infos,
                draw_block_start_circle_idxs,
                blockIdx_lin
            );
            CUDA_CHECK(cudaGetLastError());

        }
    }
    // print_gpu_mem<<<1, 1>>>(circle_infos, 0, 10);
    // CUDA_CHECK(cudaGetLastError());

    // print_gpu_mem<<<1, 1>>>(draw_block_start_circle_idxs, 0, 16);
    // CUDA_CHECK(cudaGetLastError());
}

constexpr int REG_TILE_SIZE = 3;

__device__ __forceinline__ void draw_circle(
    dim3 outIdx,
    int x_min, int x_max, int y_min, int y_max,
    CircleInfo circle_infos,
    float img_red_reg[REG_TILE_SIZE][REG_TILE_SIZE],
    float img_green_reg[REG_TILE_SIZE][REG_TILE_SIZE],
    float img_blue_reg[REG_TILE_SIZE][REG_TILE_SIZE]
) {

    const float c_x = circle_infos.cx;
    const float c_y = circle_infos.cy;
    const float c_radius = circle_infos.c_radius;
    const float c_r = circle_infos.c_r;
    const float c_g = circle_infos.c_g;
    const float c_b = circle_infos.c_b;
    const float pixel_alpha = circle_infos.pixel_alpha;

    int x_base = outIdx.x + x_min;
    int y_base = outIdx.y + y_min;

    #pragma unroll
    for (int y = 0; y < REG_TILE_SIZE; y++) {
        #pragma unroll
        for (int x = 0; x < REG_TILE_SIZE; x++) {
            int x_idx = x_base + x;
            int y_idx = y_base + y;
            float dx = x_idx - c_x;
            float dy = y_idx - c_y;
            if (dx * dx + dy * dy < c_radius * c_radius) {
                img_red_reg[y][x] = c_r * pixel_alpha + img_red_reg[y][x] * (1.0f - pixel_alpha);
                img_green_reg[y][x] = c_g * pixel_alpha + img_green_reg[y][x] * (1.0f - pixel_alpha);
                img_blue_reg[y][x] = c_b * pixel_alpha + img_blue_reg[y][x] * (1.0f - pixel_alpha);
            }
        }
    }
}


__global__ void draw_block(
    float* img_red,
    float* img_green,
    float* img_blue,
    CircleInfo* circle_infos, 
    uint32_t* draw_block_start_circle_idxs,
    dim3 draw_blockDim,
    int width,
    int height,
    int shmem_bytes) {
    
    int x_min = blockIdx.x * draw_blockDim.x;
    int x_max = (1 + blockIdx.x) * draw_blockDim.x;

    int y_min = blockIdx.y * draw_blockDim.y;
    int y_max = (1 + blockIdx.y) * draw_blockDim.y;

    // shared memory setup
    extern __shared__ CircleInfo shmem[];

    // setup register tiles
    dim3 outIdx(threadIdx.x * REG_TILE_SIZE, threadIdx.y * REG_TILE_SIZE);
    float img_red_reg[REG_TILE_SIZE][REG_TILE_SIZE];
    float img_green_reg[REG_TILE_SIZE][REG_TILE_SIZE];
    float img_blue_reg[REG_TILE_SIZE][REG_TILE_SIZE];

    // init background to white
    #pragma unroll
    for (int i = 0; i < REG_TILE_SIZE; i++) {
        #pragma unroll
        for (int j = 0; j < REG_TILE_SIZE; j++) {
            img_red_reg[i][j] = 1.0f;
            img_green_reg[i][j] = 1.0f;
            img_blue_reg[i][j] = 1.0f;
        }
    }


    int blockIdx_lin = blockIdx.y * gridDim.x + blockIdx.x;

    uint32_t start_idx = (blockIdx_lin - 1 >= 0) ? draw_block_start_circle_idxs[blockIdx_lin - 1] : 0;
    uint32_t end_idx = draw_block_start_circle_idxs[blockIdx_lin];

    // write circles_info (assumes blockDim.x >= num_circles_info)
    int threadIdx_lin = threadIdx.y * blockDim.x + threadIdx.x;
    int max_circle_infos = shmem_bytes / sizeof(CircleInfo);
    int len = end_idx - start_idx;
    for (int i = 0; i < len; i += max_circle_infos) {
        int j = i + threadIdx_lin;
        if (j < len && threadIdx_lin < max_circle_infos) {
            shmem[threadIdx_lin] = circle_infos[j + start_idx];
        }
        __syncthreads();

        int num_circles = min(len - i, max_circle_infos);
        for (int k = 0; k < num_circles; k++) {
            // printf("num_circles: %d\n", num_circles);
            draw_circle(
                outIdx,
                x_min, x_max, y_min, y_max,
                shmem[k],
                img_red_reg, img_green_reg, img_blue_reg
            );
            __syncthreads();
        }
    }

    #pragma unroll
    for (int y = 0; y < REG_TILE_SIZE; y++) {
        #pragma unroll
        for (int x = 0; x < REG_TILE_SIZE; x++) {
            int y_global = outIdx.y + y + y_min;
            int x_global = outIdx.x + x + x_min;

            if (x_global < width && y_global < height) {
                img_red[y_global * width + x_global] = img_red_reg[y][x];
                img_green[y_global * width + x_global] = img_green_reg[y][x];
                img_blue[y_global * width + x_global] = img_blue_reg[y][x];
            }
        }
    }
}


void launch_render(
    int32_t width,
    int32_t height,
    int32_t n_circle,
    float const *circle_x,      // pointer to GPU memory
    float const *circle_y,      // pointer to GPU memory
    float const *circle_radius, // pointer to GPU memory
    float const *circle_red,    // pointer to GPU memory
    float const *circle_green,  // pointer to GPU memory
    float const *circle_blue,   // pointer to GPU memory
    float const *circle_alpha,  // pointer to GPU memory
    float *img_red,             // pointer to GPU memory
    float *img_green,           // pointer to GPU memory
    float *img_blue,            // pointer to GPU memory
    GpuMemoryPool &memory_pool) {

    dim3 blockDim(REG_TILE_SIZE * 32, REG_TILE_SIZE * 32);
    dim3 num_blocks(
        ceil_div(width, blockDim.x),
        ceil_div(height, blockDim.y)
    );

    uint32_t shmem_bytes = 8192;

    CUDA_CHECK(cudaFuncSetAttribute(
        draw_block,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        shmem_bytes
    ));

    size_t num_bytes = sizeof(CircleInfo) * 4 * n_circle;
    CircleInfo* circle_infos = reinterpret_cast<CircleInfo*>(memory_pool.alloc(num_bytes));

    num_bytes = sizeof(uint32_t) * num_blocks.x * num_blocks.y;
    uint32_t* block_start_circle_idxs = reinterpret_cast<uint32_t*>(memory_pool.alloc(num_bytes));

    reorder_circles_per_block(
        n_circle, 
        circle_x, 
        circle_y, 
        circle_radius, 
        circle_red,
        circle_green,
        circle_blue,
        circle_alpha,
        memory_pool, 
        blockDim, 
        num_blocks, 
        circle_infos, 
        block_start_circle_idxs
    );

    draw_block<<<num_blocks, dim3(32, 32), shmem_bytes>>>(
        img_red,
        img_green,
        img_blue,
        circle_infos,
        block_start_circle_idxs,
        blockDim,
        width,
        height,
        shmem_bytes
    );

    CUDA_CHECK(cudaGetLastError());

}

} // namespace circles_gpu

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

GpuMemoryPool::~GpuMemoryPool() {
    for (auto ptr : allocations_) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void *GpuMemoryPool::alloc(size_t size) {
    if (next_idx_ < allocations_.size()) {
        auto idx = next_idx_++;
        if (size > capacities_.at(idx)) {
            CUDA_CHECK(cudaFree(allocations_.at(idx)));
            CUDA_CHECK(cudaMalloc(&allocations_.at(idx), size));
            CUDA_CHECK(cudaMemset(allocations_.at(idx), 0, size));
            capacities_.at(idx) = size;
        }
        return allocations_.at(idx);
    } else {
        void *ptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        CUDA_CHECK(cudaMemset(ptr, 0, size));
        allocations_.push_back(ptr);
        capacities_.push_back(size);
        next_idx_++;
        return ptr;
    }
}

void GpuMemoryPool::reset() {
    next_idx_ = 0;
    for (int32_t i = 0; i < allocations_.size(); i++) {
        CUDA_CHECK(cudaMemset(allocations_.at(i), 0, capacities_.at(i)));
    }
}

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

struct Scene {
    int32_t width;
    int32_t height;
    std::vector<float> circle_x;
    std::vector<float> circle_y;
    std::vector<float> circle_radius;
    std::vector<float> circle_red;
    std::vector<float> circle_green;
    std::vector<float> circle_blue;
    std::vector<float> circle_alpha;

    int32_t n_circle() const { return circle_x.size(); }
};

struct Image {
    int32_t width;
    int32_t height;
    std::vector<float> red;
    std::vector<float> green;
    std::vector<float> blue;
};

float max_abs_diff(Image const &a, Image const &b) {
    float max_diff = 0.0f;
    for (int32_t idx = 0; idx < a.width * a.height; idx++) {
        float diff_red = std::abs(a.red.at(idx) - b.red.at(idx));
        float diff_green = std::abs(a.green.at(idx) - b.green.at(idx));
        float diff_blue = std::abs(a.blue.at(idx) - b.blue.at(idx));
        max_diff = std::max(max_diff, diff_red);
        max_diff = std::max(max_diff, diff_green);
        max_diff = std::max(max_diff, diff_blue);
    }
    return max_diff;
}

struct Results {
    bool correct;
    float max_abs_diff;
    Image image_expected;
    Image image_actual;
    double time_ms;
};

enum class Mode {
    TEST,
    BENCHMARK,
};

template <typename T> struct GpuBuf {
    T *data;

    explicit GpuBuf(size_t n) { CUDA_CHECK(cudaMalloc(&data, n * sizeof(T))); }

    explicit GpuBuf(std::vector<T> const &host_data) {
        CUDA_CHECK(cudaMalloc(&data, host_data.size() * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(
            data,
            host_data.data(),
            host_data.size() * sizeof(T),
            cudaMemcpyHostToDevice));
    }

    ~GpuBuf() { CUDA_CHECK(cudaFree(data)); }
};

Results run_config(Mode mode, Scene const &scene) {
    auto img_expected = Image{
        scene.width,
        scene.height,
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f)};

    render_cpu(
        scene.width,
        scene.height,
        scene.n_circle(),
        scene.circle_x.data(),
        scene.circle_y.data(),
        scene.circle_radius.data(),
        scene.circle_red.data(),
        scene.circle_green.data(),
        scene.circle_blue.data(),
        scene.circle_alpha.data(),
        img_expected.red.data(),
        img_expected.green.data(),
        img_expected.blue.data());

    auto circle_x_gpu = GpuBuf<float>(scene.circle_x);
    auto circle_y_gpu = GpuBuf<float>(scene.circle_y);
    auto circle_radius_gpu = GpuBuf<float>(scene.circle_radius);
    auto circle_red_gpu = GpuBuf<float>(scene.circle_red);
    auto circle_green_gpu = GpuBuf<float>(scene.circle_green);
    auto circle_blue_gpu = GpuBuf<float>(scene.circle_blue);
    auto circle_alpha_gpu = GpuBuf<float>(scene.circle_alpha);
    auto img_red_gpu = GpuBuf<float>(scene.height * scene.width);
    auto img_green_gpu = GpuBuf<float>(scene.height * scene.width);
    auto img_blue_gpu = GpuBuf<float>(scene.height * scene.width);

    auto memory_pool = GpuMemoryPool();

    auto reset = [&]() {
        CUDA_CHECK(
            cudaMemset(img_red_gpu.data, 0, scene.height * scene.width * sizeof(float)));
        CUDA_CHECK(cudaMemset(
            img_green_gpu.data,
            0,
            scene.height * scene.width * sizeof(float)));
        CUDA_CHECK(
            cudaMemset(img_blue_gpu.data, 0, scene.height * scene.width * sizeof(float)));
        memory_pool.reset();
    };

    auto f = [&]() {
        circles_gpu::launch_render(
            scene.width,
            scene.height,
            scene.n_circle(),
            circle_x_gpu.data,
            circle_y_gpu.data,
            circle_radius_gpu.data,
            circle_red_gpu.data,
            circle_green_gpu.data,
            circle_blue_gpu.data,
            circle_alpha_gpu.data,
            img_red_gpu.data,
            img_green_gpu.data,
            img_blue_gpu.data,
            memory_pool);
    };

    reset();
    f();

    auto img_actual = Image{
        scene.width,
        scene.height,
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f)};

    CUDA_CHECK(cudaMemcpy(
        img_actual.red.data(),
        img_red_gpu.data,
        scene.height * scene.width * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        img_actual.green.data(),
        img_green_gpu.data,
        scene.height * scene.width * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        img_actual.blue.data(),
        img_blue_gpu.data,
        scene.height * scene.width * sizeof(float),
        cudaMemcpyDeviceToHost));

    float max_diff = max_abs_diff(img_expected, img_actual);

    if (max_diff > 1e-3) {
        return Results{
            false,
            max_diff,
            std::move(img_expected),
            std::move(img_actual),
            0.0,
        };
    }

    if (mode == Mode::TEST) {
        return Results{
            true,
            max_diff,
            std::move(img_expected),
            std::move(img_actual),
            0.0,
        };
    }

    double time_ms = benchmark_ms(1000.0, reset, f);

    return Results{
        true,
        max_diff,
        std::move(img_expected),
        std::move(img_actual),
        time_ms,
    };
}

template <typename Rng>
Scene gen_random(Rng &rng, int32_t width, int32_t height, int32_t n_circle) {
    auto unif_0_1 = std::uniform_real_distribution<float>(0.0f, 1.0f);
    auto z_values = std::vector<float>();
    for (int32_t i = 0; i < n_circle; i++) {
        float z;
        for (;;) {
            z = unif_0_1(rng);
            z = std::max(z, unif_0_1(rng));
            if (z > 0.01) {
                break;
            }
        }
        // float z = std::max(unif_0_1(rng), unif_0_1(rng));
        z_values.push_back(z);
    }
    std::sort(z_values.begin(), z_values.end(), std::greater<float>());

    auto colors = std::vector<uint32_t>{
        0xd32360,
        0xcc9f26,
        0x208020,
        0x2874aa,
    };
    auto color_idx_dist = std::uniform_int_distribution<int>(0, colors.size() - 1);
    auto alpha_dist = std::uniform_real_distribution<float>(0.0f, 0.5f);

    int32_t fog_interval = n_circle / 10;
    float fog_alpha = 0.2;

    auto scene = Scene{width, height};
    float base_radius_scale = 1.0f;
    int32_t i = 0;
    for (float z : z_values) {
        float max_radius = base_radius_scale / z;
        float radius = std::max(1.0f, unif_0_1(rng) * max_radius);
        float x = unif_0_1(rng) * (width + 2 * max_radius) - max_radius;
        float y = unif_0_1(rng) * (height + 2 * max_radius) - max_radius;
        int color_idx = color_idx_dist(rng);
        uint32_t color = colors[color_idx];
        scene.circle_x.push_back(x);
        scene.circle_y.push_back(y);
        scene.circle_radius.push_back(radius);
        scene.circle_red.push_back(float((color >> 16) & 0xff) / 255.0f);
        scene.circle_green.push_back(float((color >> 8) & 0xff) / 255.0f);
        scene.circle_blue.push_back(float(color & 0xff) / 255.0f);
        scene.circle_alpha.push_back(alpha_dist(rng));
        i++;
        if (i % fog_interval == 0 && i + 1 < n_circle) {
            scene.circle_x.push_back(float(width - 1) / 2.0f);
            scene.circle_y.push_back(float(height - 1) / 2.0f);
            scene.circle_radius.push_back(float(std::max(width, height)));
            scene.circle_red.push_back(1.0f);
            scene.circle_green.push_back(1.0f);
            scene.circle_blue.push_back(1.0f);
            scene.circle_alpha.push_back(fog_alpha);
        }
    }

    return scene;
}

constexpr float PI = 3.14159265359f;

Scene gen_overlapping_opaque() {
    int32_t width = 256;
    int32_t height = 256;

    auto scene = Scene{width, height};

    auto colors = std::vector<uint32_t>{
        0xd32360,
        0xcc9f26,
        0x208020,
        0x2874aa,
    };

    int32_t n_circle = 20;
    int32_t n_ring = 4;
    float angle_range = PI;
    for (int32_t ring = 0; ring < n_ring; ring++) {
        float dist = 20.0f * (ring + 1);
        float saturation = float(ring + 1) / n_ring;
        float hue_shift = float(ring) / (n_ring - 1);
        for (int32_t i = 0; i < n_circle; i++) {
            float theta = angle_range * i / (n_circle - 1);
            float x = width / 2.0f - dist * std::cos(theta);
            float y = height / 2.0f - dist * std::sin(theta);
            scene.circle_x.push_back(x);
            scene.circle_y.push_back(y);
            scene.circle_radius.push_back(16.0f);
            auto color = colors[(i + ring * 2) % colors.size()];
            scene.circle_red.push_back(float((color >> 16) & 0xff) / 255.0f);
            scene.circle_green.push_back(float((color >> 8) & 0xff) / 255.0f);
            scene.circle_blue.push_back(float(color & 0xff) / 255.0f);
            scene.circle_alpha.push_back(1.0f);
        }
    }

    return scene;
}

Scene gen_overlapping_transparent() {
    int32_t width = 256;
    int32_t height = 256;

    auto scene = Scene{width, height};

    float offset = 20.0f;
    float radius = 40.0f;
    scene.circle_x = std::vector<float>{
        (width - 1) / 2.0f - offset,
        (width - 1) / 2.0f + offset,
        (width - 1) / 2.0f + offset,
        (width - 1) / 2.0f - offset,
    };
    scene.circle_y = std::vector<float>{
        (height - 1) * 0.75f,
        (height - 1) * 0.75f,
        (height - 1) * 0.25f,
        (height - 1) * 0.25f,
    };
    scene.circle_radius = std::vector<float>{
        radius,
        radius,
        radius,
        radius,
    };
    // 0xd32360
    // 0x2874aa
    scene.circle_red = std::vector<float>{
        float(0xd3) / 255.0f,
        float(0x28) / 255.0f,
        float(0x28) / 255.0f,
        float(0xd3) / 255.0f,
    };
    scene.circle_green = std::vector<float>{
        float(0x23) / 255.0f,
        float(0x74) / 255.0f,
        float(0x74) / 255.0f,
        float(0x23) / 255.0f,
    };
    scene.circle_blue = std::vector<float>{
        float(0x60) / 255.0f,
        float(0xaa) / 255.0f,
        float(0xaa) / 255.0f,
        float(0x60) / 255.0f,
    };
    scene.circle_alpha = std::vector<float>{
        0.75f,
        0.75f,
        0.75f,
        0.75f,
    };
    return scene;
}

Scene gen_simple() {
    /*
        0xd32360,
        0xcc9f26,
        0x208020,
        0x2874aa,
    */
    int32_t width = 256;
    int32_t height = 256;
    auto scene = Scene{width, height};
    scene.circle_x = std::vector<float>{
        (width - 1) * 0.25f,
        (width - 1) * 0.75f,
        (width - 1) * 0.25f,
        (width - 1) * 0.75f,
    };
    scene.circle_y = std::vector<float>{
        (height - 1) * 0.25f,
        (height - 1) * 0.25f,
        (height - 1) * 0.75f,
        (height - 1) * 0.75f,
    };
    scene.circle_radius = std::vector<float>{
        40.0f,
        40.0f,
        40.0f,
        40.0f,
    };
    scene.circle_red = std::vector<float>{
        float(0xd3) / 255.0f,
        float(0xcc) / 255.0f,
        float(0x20) / 255.0f,
        float(0x28) / 255.0f,
    };
    scene.circle_green = std::vector<float>{
        float(0x23) / 255.0f,
        float(0x9f) / 255.0f,
        float(0x80) / 255.0f,
        float(0x74) / 255.0f,
    };
    scene.circle_blue = std::vector<float>{
        float(0x60) / 255.0f,
        float(0x26) / 255.0f,
        float(0x20) / 255.0f,
        float(0xaa) / 255.0f,
    };
    scene.circle_alpha = std::vector<float>{
        1.0f,
        1.0f,
        1.0f,
        1.0f,
    };
    return scene;
}

// Output image writers: BMP file header structure
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t fileType{0x4D42};   // File type, always "BM"
    uint32_t fileSize{0};        // Size of the file in bytes
    uint16_t reserved1{0};       // Always 0
    uint16_t reserved2{0};       // Always 0
    uint32_t dataOffset{54};     // Start position of pixel data
    uint32_t headerSize{40};     // Size of this header (40 bytes)
    int32_t width{0};            // Image width in pixels
    int32_t height{0};           // Image height in pixels
    uint16_t planes{1};          // Number of color planes
    uint16_t bitsPerPixel{24};   // Bits per pixel (24 for RGB)
    uint32_t compression{0};     // Compression method (0 for uncompressed)
    uint32_t imageSize{0};       // Size of raw bitmap data
    int32_t xPixelsPerMeter{0};  // Horizontal resolution
    int32_t yPixelsPerMeter{0};  // Vertical resolution
    uint32_t colorsUsed{0};      // Number of colors in the color palette
    uint32_t importantColors{0}; // Number of important colors
};
#pragma pack(pop)

void write_bmp(
    std::string const &fname,
    uint32_t width,
    uint32_t height,
    const std::vector<uint8_t> &pixels) {
    BMPHeader header;
    header.width = width;
    header.height = height;

    uint32_t rowSize = (width * 3 + 3) & (~3); // Align to 4 bytes
    header.imageSize = rowSize * height;
    header.fileSize = header.dataOffset + header.imageSize;

    std::ofstream file(fname, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));

    // Write pixel data with padding
    std::vector<uint8_t> padding(rowSize - width * 3, 0);
    for (int32_t idx_y = height - 1; idx_y >= 0;
         --idx_y) { // BMP stores pixels from bottom to top
        const uint8_t *row = &pixels[idx_y * width * 3];
        file.write(reinterpret_cast<const char *>(row), width * 3);
        if (!padding.empty()) {
            file.write(reinterpret_cast<const char *>(padding.data()), padding.size());
        }
    }
}

uint8_t float_to_byte(float x) {
    if (x < 0) {
        return 0;
    } else if (x >= 1) {
        return 255;
    } else {
        return x * 255.0f;
    }
}

void write_image(std::string const &fname, Image const &img) {
    auto pixels = std::vector<uint8_t>(img.width * img.height * 3);
    for (int32_t idx = 0; idx < img.width * img.height; idx++) {
        float red = img.red.at(idx);
        float green = img.green.at(idx);
        float blue = img.blue.at(idx);
        // BMP stores pixels in BGR order
        pixels.at(idx * 3) = float_to_byte(blue);
        pixels.at(idx * 3 + 1) = float_to_byte(green);
        pixels.at(idx * 3 + 2) = float_to_byte(red);
    }
    write_bmp(fname, img.width, img.height, pixels);
}

Image compute_img_diff(Image const &a, Image const &b) {
    auto img_diff = Image{
        a.width,
        a.height,
        std::vector<float>(a.height * a.width, 0.0f),
        std::vector<float>(a.height * a.width, 0.0f),
        std::vector<float>(a.height * a.width, 0.0f),
    };
    for (int32_t idx = 0; idx < a.width * a.height; idx++) {
        img_diff.red.at(idx) = std::abs(a.red.at(idx) - b.red.at(idx));
        img_diff.green.at(idx) = std::abs(a.green.at(idx) - b.green.at(idx));
        img_diff.blue.at(idx) = std::abs(a.blue.at(idx) - b.blue.at(idx));
    }
    return img_diff;
}

struct SceneTest {
    std::string name;
    Mode mode;
    Scene scene;
};

int main(int argc, char const *const *argv) {
    auto rng = std::mt19937(0xCA7CAFE);

    auto scenes = std::vector<SceneTest>();
    scenes.push_back({"simple", Mode::TEST, gen_simple()});
    scenes.push_back({"overlapping_opaque", Mode::TEST, gen_overlapping_opaque()});
    scenes.push_back(
        {"overlapping_transparent", Mode::TEST, gen_overlapping_transparent()});
    scenes.push_back(
        {"million_circles", Mode::BENCHMARK, gen_random(rng, 1024, 1024, 1'000'000)});

    int32_t fail_count = 0;

    int32_t count = 0;
    for (auto const &scene_test : scenes) {
        auto i = count++;
        printf("\nTesting scene '%s'\n", scene_test.name.c_str());
        auto results = run_config(scene_test.mode, scene_test.scene);
        write_image(
            std::string("out/img") + std::to_string(i) + "_" + scene_test.name +
                "_cpu.bmp",
            results.image_expected);
        write_image(
            std::string("out/img") + std::to_string(i) + "_" + scene_test.name +
                "_gpu.bmp",
            results.image_actual);
        if (!results.correct) {
            printf("  Result did not match expected image\n");
            printf("  Max absolute difference: %.2e\n", results.max_abs_diff);
            auto diff = compute_img_diff(results.image_expected, results.image_actual);
            write_image(
                std::string("out/img") + std::to_string(i) + "_" + scene_test.name +
                    "_diff.bmp",
                diff);
            printf(
                "  (Wrote image diff to 'out/img%d_%s_diff.bmp')\n",
                i,
                scene_test.name.c_str());
            fail_count++;
            continue;
        } else {
            printf("  OK\n");
        }
        if (scene_test.mode == Mode::BENCHMARK) {
            printf("  Time: %f ms\n", results.time_ms);
        }
    }

    if (fail_count) {
        printf("\nCorrectness: %d tests failed\n", fail_count);
    } else {
        printf("\nCorrectness: All tests passed\n");
    }

    return 0;
}
