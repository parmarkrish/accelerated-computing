### Question 1

The best runtime I achieved was 6.22 ms. This is 9.33 TFlops. 
Since the max theoretical TFLOPs on an a4000 is 19.1 TFLOPs. I achieve ~49% FLOPs utilization.

I used the following techniques:
- Overlapping data movement with computation with async copy's
- Vectorized loads to shared memory (part of async copy's)
- Register tiling (4x4)

Increasing register tile size delivered a massive performance improvement as L1 to register memory was no longer a bottleneck.
This performance improvement only resulted by increasing MICROTILE_REDUCE_DIM. If MICROTILE_REDUCE_DIM = 1, there would be no performance improvement.

I believe the reason for this is because increasing MICROTILE_REDUCE_DIM resulted in better coalesed memory transactions from L1 to registers. 

- For some reason adding -maxrregcount=64 improves performance.

### Question 2

1. 

| problem size       | GFLOPS 
|--------------------|-------
| (3072, 3072, 3072) | 57.98 
| (512, 3072, 3072)  | 9.66 
| (256, 3072, 3072)  | 4.83 
| (128, 3072, 3072)  | 2.42 
| (64, 3072, 3072)   | 1.21 
| (32, 3072, 3072)   | 0.60 
| (16, 3072, 3072)   | 0.30 
| (1, 3072, 3072)    | 0.02 
| (256, 256, 256)    | 0.03 
| (256, 256, 1024)   | 0.13 
| (256, 256, 8192)   | 1.07 
| (128, 128, 32768)  | 1.07 


2. Considering the 19.1 max theoretical TFLOPs of an a4000

| problem size       | time (ms) 
|--------------------|-------
| (3072, 3072, 3072) | 3.04
| (512, 3072, 3072)  | 0.51
| (256, 3072, 3072)  | 0.25
| (128, 3072, 3072)  | 0.13
| (64, 3072, 3072)   | 0.06
| (32, 3072, 3072)   | 0.03
| (16, 3072, 3072)   | 0.02
| (1, 3072, 3072)    | 0.001
| (256, 256, 256)    | 0.002
| (256, 256, 1024)   | 0.07
| (256, 256, 8192)   | 0.06
| (128, 128, 32768)  | 0.06

3. 

| problem size       | size (MB) |
|--------------------|-------:|
| (3072, 3072, 3072) |   113
| (512, 3072, 3072)  |  50.3
| (256, 3072, 3072)  |  44.0
| (128, 3072, 3072)  |  40.9
| (64, 3072, 3072)   |  39.3
| (32, 3072, 3072)   |  38.5
| (16, 3072, 3072)   |  38.1
| (1, 3072, 3072)    |  37.8
| (256, 256, 256)    | 0.786
| (256, 256, 1024)   |  2.36
| (256, 256, 8192)   |  17.0
| (128, 128, 32768)  |  33.6

4. The a4000 has a memory bandwidth of 448 GB / sec

| problem size       | time (ms) |
|--------------------|-------:|
| (3072, 3072, 3072) |  0.253 
| (512, 3072, 3072)  |  0.112 
| (256, 3072, 3072)  | 0.0983 
| (128, 3072, 3072)  | 0.0913 
| (64, 3072, 3072)   | 0.0878 
| (32, 3072, 3072)   | 0.0860 
| (16, 3072, 3072)   | 0.0851 
| (1, 3072, 3072)    | 0.0843 
| (256, 256, 256)    | 0.00176
| (256, 256, 1024)   | 0.00527
| (256, 256, 8192)   | 0.0380 
| (128, 128, 32768)  | 0.0750 

5. 

| problem size       | Bound
|--------------------|-------:|
| (3072, 3072, 3072) | compute 
| (512, 3072, 3072)  | compute 
| (256, 3072, 3072)  | compute 
| (128, 3072, 3072)  | compute 
| (64, 3072, 3072)   | memory-bandwidth 
| (32, 3072, 3072)   | memory-bandwith 
| (16, 3072, 3072)   | memory-bandwith 
| (1, 3072, 3072)    | memory-bandwith 
| (256, 256, 256)    | compute 
| (256, 256, 1024)   | compute 
| (256, 256, 8192)   | compute 
| (128, 128, 32768)  | memory-bandwith 

6. 

| problem size       | Max theoretical TFLOP/s |
|--------------------|-----------------:|
| (3072, 3072, 3072) | 19.1
| (512, 3072, 3072)  | 19.1
| (256, 3072, 3072)  | 19.1 
| (128, 3072, 3072)  | 19.1 
| (64, 3072, 3072)   | 13.8 
| (32, 3072, 3072)   | 7.02 
| (16, 3072, 3072)   | 3.54 
| (1, 3072, 3072)    | 0.22 
| (256, 256, 256)    | 19.1 
| (256, 256, 1024)   | 19.1 
| (256, 256, 8192)   | 19.1 
| (128, 128, 32768)  | 14.3 

7. 

| problem size       | # of thread blocks |
|--------------------|-------------------:|
| (3072, 3072, 3072) | 576 
| (512, 3072, 3072)  | 96 
| (256, 3072, 3072)  | 48 
| (128, 3072, 3072)  | 24
| (64, 3072, 3072)   | 24
| (32, 3072, 3072)   | 24
| (16, 3072, 3072)   | 24
| (1, 3072, 3072)    | 24
| (256, 256, 256)    | 4
| (256, 256, 1024)   | 4
| (256, 256, 8192)   | 4
| (128, 128, 32768)  | 1

8. An a4000 has 48 SMs

| problem size       | More or less Threadblocks than SMs
|--------------------|----------------------------------
| (3072, 3072, 3072) | More
| (512, 3072, 3072)  | More
| (256, 3072, 3072)  | More
| (128, 3072, 3072)  | Less
| (64, 3072, 3072)   | Less
| (32, 3072, 3072)   | Less
| (16, 3072, 3072)   | Less
| (1, 3072, 3072)    | Less
| (256, 256, 256)    | Less
| (256, 256, 1024)   | Less
| (256, 256, 8192)   | Less
| (128, 128, 32768)  | Less

9. Having less threadblocks than SMs, reduces both memory bandwidth and computation power. 
So, for problem sizes where we have less threadblocks than SMs the actual TFLOPs would be quite a bit lower than the expected TFLOPs in 6.

10. For problem sizes that have more threadblocks than SMs i'm obtaining ~40% FLOP utilization. However if the problem size has much less threadblocks than SMs, the performance decreases dramatically. 

| problem size       | TFLOP/s |
|--------------------|-----------------:|
| (3072, 3072, 3072) | 7.39
| (512, 3072, 3072)  | 8.05
| (256, 3072, 3072)  | 9.19
| (128, 3072, 3072)  | 4.63
| (64, 3072, 3072)   | 2.42
| (32, 3072, 3072)   | 1.23
| (16, 3072, 3072)   | 0.62
| (1, 3072, 3072)    | 0.04
| (256, 256, 256)    | 0.74
| (256, 256, 1024)   | 0.83
| (256, 256, 8192)   | 0.86
| (128, 128, 32768)  | 0.22

### Question 3

| problem size       | TFLOP/s | % of max theoretical
| -------------------|---------|---------------------
| (3072, 3072, 3072) | 6.20    |      32.46% 
| (512, 3072, 3072)  | 5.98    |      31.31% 
| (256, 3072, 3072)  | 5.87    |      30.73% 
| (128, 3072, 3072)  | 5.83    |      30.52% 
| (64, 3072, 3072)   | 3.29    |      23.84% 
| (32, 3072, 3072)   | 1.75    |      24.93% 
| (16, 3072, 3072)   | 0.90    |      25.42% 
| (1, 3072, 3072)    | 0.06    |      27.27% 
| (256, 256, 256)    | 0.24    |       1.26% 
| (256, 256, 1024)   | 0.95    |       4.97% 
| (256, 256, 8192)   | 5.62    |      29.42% 
| (128, 128, 32768)  | 3.86    |      26.99% 

- `matmul_improve_reduce` has slighly reduced performance for large square-ish matmuls. But when `size_i` and `size_j` are small and `size_k` is large, it has significantly increased performance

- I set tuned the `K_SPLIT_SIZE` parameter by starting at 3072 and decreasing by a factor of 2 such that all problem sizes except (256, 256, 256) and (256, 256, 1024) were above 20%.

- For (256, 256, 256) and (256, 256, 1024) problem sizes I achieved 1.26% and 4.97% max theoretical throughput. For those problem sizes, there simply isn't enough work to fill up all the SMs, even when performing split-k. 

- 



