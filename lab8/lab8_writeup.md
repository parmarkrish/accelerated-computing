### Question 1

I was able to achieve a run-time of 31.58 ms.

My solution has two phases. 

In the first phase, we parallelize over circles. For every block, stream compaction is used to identify the circles that intersect with the block and then the respective circle information is stored in render order. All the circle info for all the blocks is stored contiguously in a buffer called `circle_infos`. Starting indices for each block into the buffer are stored in `block_start_circle_idxs`. In order to implement stream compaction I modified my scan implementation from lab7. I also don't store the boolean mask as I recompute it by calculating changes in the scan output.

In the second phase, we draw a block by retrieving all the circle info for each block and drawing each circle consequently. Since the circle information is stored in render order, the alpha blending is equivalent to the CPU version. We parallelize over pixels of a block here. Each thread is responsible for a 3x3 output region of each block. The channels of the output region are stored in registers for fast access since we are reading and writing it for every circle that intersects with the block. 

An earlier version of this phase parallelized over each circle region instead of the block background and this was much slower (~70 ms) because sometimes the circle would be much larger than the block and most threads would remain inactive due to masking for a long period of time.
