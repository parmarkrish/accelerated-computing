### Question 1

1. We are doing 3072 FMA's for element in output matrix.
So we are doing 3072 * 3072 * (2 * 3072) = ~58 GFLOPs in total.

2. An RTX A4000 has a max FP performance of 19.17 TFLOPs.
So it would take 58 GFLOPS / 19.17 TFLOPS = 3.02 ms.

3. For A and B matrices we each read 3072 x 3072 locations and write 3072 x 3072 locations to C. In total, access 3 * 3072 * 3072 = 28.3M memory locations. Or 28.3M * 4 FLOPs/location = 113.2 MB of DRAM read/write

4. Memory bandwidth for an A4000 is 448 GB/s. So it would take 113.2 MB / 448 GB/s = 0.25 ms.

5. (4) is significantly less than (2). Even if each location is access 12 times, we would still be compute bound.
So I would expect the run time of this workload to be dominated by compute (given a well optimized implementation).

6. We would load in 3072 * 3072 * (3072 * 2 * 4) = 231.9 GB of loads from DRAM.

7. It would take 231.9 GB / 448 GB/s = 517.70 ms

8. The L2 memory bandwidth on an L2 is 1.5 TB/sec. So it would take 231.9 GB / 1.5 TB/sec = 154.62 ms.

9. (8) is ~3.3 times faster than (7).


### Question 2

I was able to achieve a runtime of 55.83 ms. My output tile size is 32 x 32. I choose the largest tile size the hardware can support for which each thread handles one output element. Tile size influences the amount of memory reuse in the kernel. For a tile size of K x K, each thread does K times less loads, hence why I choose the largest tile-size supported by the hardware (maximum block-size when each thread is responsible for a single output element). 

As for the algorithm, I used block matrix multiplication where the A & B blocks are loaded into shared memory to compute partial C blocks.


### Question 3

As in Q1-6, there would be 231.9 GB of loads. So it would take 231.9 GB / 9.6 TB/sec = 24.16 ms.


### Question 4

I was able to achieve a runtime of 13.28 ms.
Each thread is responsible for a K x K block of output element. Each thread loads in K x K blocks A and B into shared memory, so the tile_size in shared memory is (block_size.x * K, block_size.y * K). 

Then we have another level of block matrix multiply were each thread loads in K x K tiles from shared memory into the registers thereby leverering register level reuse.

I did need to set the value of K. K=2 got the fastest runtime of 13.28 ms. Meanwhile K=3 achieved 16.80 ms and K=4 resulted in an error due to insufficient shared memory.

AFAIK, there are no bank conflicts.

When I inspect the generated SASS, I see that FFMA instructions do share some registers:

<pre>
/*0840*/    FFMA.FTZ R9,  R13.reuse, R24.reuse, R33 ;      /* 0x000000180d097223 */
                                                        /* 0x0c4fe20000010021 */
/*0850*/    FFMA.FTZ R12, R13,       R25.reuse, R12 ;     /* 0x000000190d0c7223 */
                                                        /* 0x080fe2000001000c */
/*0860*/    FFMA.FTZ R33, R11.reuse, R24,       R8 ;      /* 0x000000180b217223 */
                                                        /* 0x040fe20000010008 */
/*0870*/    FFMA.FTZ R34, R11,       R25,       R10 ;     /* 0x000000190b227223 */
                                                        /* 0x000fe2000001000a */
/*0880*/    FFMA.FTZ R24, R14.reuse, R22,       R9 ;      /* 0x000000160e187223 */
                                                        /* 0x048fe20000010009 */
/*0890*/    FFMA.FTZ R14, R14,       R23,       R12 ;     /* 0x000000170e0e7223 */
</pre>
