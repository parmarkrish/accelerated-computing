Note: This lab was conducted using the L4 instance on Colab Pro

### Question 1

1. The total size of the u0 and u1 buffers is ~20MB (1601 * 1601 * 4 * 2). The capacity of the L2 cache in an RTX A4000 GPU is 4 MB. However, a L4 has a 48 MB L2 cache.

2. Each CUDA thread is requesting 7 float loads/stores from memory. There are 1601 * 1601 active threads. So in total 71MB (7 * 1601 * 1601 * 4) is loaded/stored from the L2 cache.

3. For the first launch, ~20MB are gotten out of DRAM, but afterwards none of the requests miss since the L2 Cache fits the entire buffer. If the GPU is an RTX A4000, if we assume perfect reuse, there would be ~20MB worth of request misses.

4. On an RTX A4000 GPU, assuming perfect reuse, it would take (20 * 1e6 / (448* 1e9) * 1e3) = 44.65 microseconds for each timestep. Multiplying that by 6400 timesteps, we obtain ~285 ms. On the other hand, if we assume no reuse, it would take ~1014 ms.

5. Doing 71 MB of loads/stores out of L2 at a memory bandwidth of 1.5TB / sec takes 47.3 microseconds for each timestep. If we multiply that by 6400 timesteps, we obtain ~302 milliseconds

6. On an L4, I obtain a maximum speed of 191 ms. Disabling L1 caching (by adding compiler flags (-Xptxas -dlcm=cg -Xptxas -dscm=cg), the speed is 282 ms. This is similar to (5) which make sense because the L4 has a 48 MB L2 cache which can comfortably fit the buffers. These buffers get reused every kernel invocation wave_gpu_naive_step. 

7. We can exploit reuse at the level of L1 cache / shared memory to speed up the kernel

### Question 2
On an L4 GPU, my shared memory kernel was actually a bit slower than the naive implementation. (215 ms vs 191 ms). The naive GPU implementation is quite good because of automatic L1 caching. 
Even then, I believe I should be able to achieve some speed up due to the shared memory implementation being multistep (L1 cached is reused over multiple steps). Strange, might come back to this later.

One interesting tradeoff is that as we increase n_steps, we can do more computation in L1 cache, but we also disable more threads because the "valid region" shrinks. My solution found that n_steps=5 was optimal.

Initially I forgot to utilize the extra buffers I would write directly to u0 and u1. This would mess up the computation of other blocks since there was some overlap. The bug was non-deterministic because I believe the order that the blocks run on the GPU is non-deterministic.

