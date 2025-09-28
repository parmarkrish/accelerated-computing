### Question 1

- The PTX for the lop3 instruction looks like `lop3.b32 %r1, %r2, %r3, %r4, 0b11101010;`. The holes got replaced with u32 PTX registers.
- The generated SASS looks like `LOP3.LUT R11, R2, R5, R6, 0xea, !PT`

### Question 2

The SASS instruction looks like `HMMA.1688.F32.TF32 R8, R8, R12, R16`

### Question 3

1. 
| problem size       | time (ms) 
|--------------------|---------
| (3072, 3072, 3072) | 1.51
| (512, 3072, 3072)  | 0.252
| (256, 3072, 3072)  | 0.126
| (128, 3072, 3072)  | 0.0630
| (64, 3072, 3072)   | 0.0315
| (32, 3072, 3072)   | 0.0158
| (16, 3072, 3072)   | 0.00788

2.

| problem size       | boundedness      | time (ms)
|--------------------|------------------|----------
| (3072, 3072, 3072) | compute          | 1.51
| (512, 3072, 3072)  | compute          | 0.252
| (256, 3072, 3072)  | compute          | 0.126
| (128, 3072, 3072)  | memory-bandwidth | 0.0913
| (64, 3072, 3072)   | memory-bandwidth | 0.0878
| (32, 3072, 3072)   | memory-bandwidth | 0.0860
| (16, 3072, 3072)   | memory-bandwidth | 0.0851

3. 

| problem size       | boundedness      | Theoretical Peak TFLOP/s
|--------------------|------------------|------------------------
| (3072, 3072, 3072) | compute          | 38.34
| (512, 3072, 3072)  | compute          | 38.34
| (256, 3072, 3072)  | compute          | 38.34
| (128, 3072, 3072)  | memory-bandwidth | 26.47
| (64, 3072, 3072)   | memory-bandwidth | 13.76
| (32, 3072, 3072)   | memory-bandwidth | 7.022
| (16, 3072, 3072)   | memory-bandwidth | 3.55

4. Almost the same. Except that problem size (128, 3072, 3072) was compute bound when using FMAs


### Question 4

| problem size       | time (ms) | TFLOP/s | Speedup compared to cublas_fma | Fraction of Peak Theoretical Throughput 
|--------------------|-----------|---------|--------------------------------|----------------------------------------
| (3072, 3072, 3072) | 3.44      | 16.86   | 1.18x                          | 43.97%
| (512, 3072, 3072)  | 0.58      | 16.58   | 1.37x                          | 43.24%
| (256, 3072, 3072)  | 0.31      | 15.67   | 1.49x                          | 40.87%
| (128, 3072, 3072)  | 0.15      | 15.76   | 1.57x                          | 59.54%
| (64, 3072, 3072)   | 0.14      | 8.71    | 0.94x                          | 63.30%
| (32, 3072, 3072)   | 0.13      | 4.57    | 0.83x                          | 65.08%
| (16, 3072, 3072)   | 0.13      | 2.37    | 0.86x                          | 66.76%

- I need the kernel to be "warp based" which meant I had to compute the warp index and base computation around that. I also needed to change the way I do register tiling because if the non-square shape of the tensor core matmul and global memory coalescing considerations.
- I observed an RRMSE of 7.74e-04 compared to a ~1.03e-06 RMSE of the non-tensor-core implementation






