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

| problem size       | boundedness      | Theoretical Max TFLOP/s
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



