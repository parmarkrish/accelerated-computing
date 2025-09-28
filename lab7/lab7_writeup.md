### Question 1

- I was able to achieve a throughput of 149.49 GB/s
- My first kernel `scan_block` performs a per block scan where each block is of size 12,288 (1024 threads per block and 12 elements per thread). Additionally, following the hierarchical scan approach, if our kernel was launched with more than one block, we store the endpoint of each block and perform a scan on these endpoints using `scan_block` again. Lastly we fix up the values using the `fix_up` kernel.
- While storing the endpoints contiguously is not necessary, I ended up doing so that I can (1) reuse `scan_block` and (2) the memory accesses would be coalesced.
- I was about to pipeline `scan_block` when I noticed that using `cp_async4` to load in data improves performance by about ~20% and I'm unsure why.


### Question 2
- I was able to achieve a runtime of 0.78 ms.
- My Design: I used a modified scan implementation as it allows me minimize global memory data movement. In `scan_block`, after loading in the raw values to shared memory, I compute the boundary indicators and replace the raw values in shared memory with them. As a result, the scan will be performed on the boundary indicators which will result in an array that gives us the index location of the respective raw data and length for the output arrays. In the `fix_up` kernel, after we have computed the `output_idxs` (result of scan), we store these values in registers and perform a per thread `rle_compress`, storing the data and lengths into their index location as given by `output_idxs`.
- Interesting bug: the `if (j > n) return` is quite important in `rle_compress`. Omitting this line potentially results in an incorrect run length value for the last element in raw data. This is because, if there are inactive threads, then these threads still atomicAdd into `compressed_lengths[output_idx]`. For these inactive threads the output_idx corresponds with the index location of the last element in raw_data, hence the run_length value for the last element may be greater higher than it should be.