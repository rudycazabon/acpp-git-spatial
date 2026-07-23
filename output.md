# Executing: ./sycl/build/version1
Allocating test payloads for 1000000 structural entities...
Target Device (CPU Backend): AdaptiveCpp OpenMP host device

--- Starting Native FP64 Execution Pass ---
FP64 Pipeline Finished in 79.1279 ms.
Sample Output 0 -> X: 1.00187e+06, Y: 5.64122e+06

--- Starting Camera-Relative FP32 Execution Pass ---
FP32 Relative Pipeline Finished in 6.28181 ms.
Sample Output 0 -> X: 500200, Y: 5.004e+06
# Executing: ./sycl/build/version2
Allocating test payloads for 1000000 structural entities...
Target Device (CPU Backend): AdaptiveCpp OpenMP host device

--- Starting Native FP64 Execution Pass ---
FP64 Pipeline Finished in 100.735 ms.
Sample Output 0 -> X: 1.00187e+06, Y: 5.64122e+06

--- Starting Camera-Relative FP32 Execution Pass ---
FP32 Relative Pipeline Finished in 33.6866 ms.
Sample Output 0 -> X: 500200, Y: 5.004e+06
# Executing: ./sycl/build/versionLast
[Harness Initialization] Selecting System Target Context...
Using Device Backend: AdaptiveCpp OpenMP host device

Executing Loops via Standard Buffer Mapping Scopes...
All kernels passed execution bounds successfully!
Benchmark Duration: 73.9595 ms.
Output 0 Sample Coordinates -> X: 500200, Y: 5.004e+06
# Executing: ./sycl/build/versionUsmAlloc2
Initializing SYCL Selection Context...
Running on Device Target: AdaptiveCpp OpenMP host device

Allocating Host-Mapped Unified Memory allocations...
Populating payload buffers with 1,000,000 entities...
Payload generation complete.

--- Starting Native Unified FP64 Execution Pass ---
-> FP64 Pipeline Finished in 89.0502 ms.
-> Sample Output 0 -> X: 1.00187e+06, Y: 5.64122e+06

--- Starting Camera-Relative Unified FP32 Execution Pass ---
-> FP32 Relative Pipeline Finished in 10.9372 ms.
-> Sample Output 0 -> X: 500200, Y: 5.004e+06

Cleaning down active execution tracking memory...
Harness process exited successfully.
# Executing: ./sycl/build/versionUsmAlloc1
Initializing SYCL Queue Context and Shared USM Allocations...
GPU context requested failed, initializing fallback Host CPU selector context...
Running on Active Hardware Instance: AdaptiveCpp OpenMP host device

--- Starting Native USM FP64 Execution Pass ---
USM FP64 Pipeline Finished in 91.0453 ms.
Sample Output 0 -> X: 1.00187e+06, Y: 5.64122e+06

--- Starting Camera-Relative USM FP32 Execution Pass ---
USM FP32 Relative Pipeline Finished in 7.96012 ms.
Sample Output 0 -> X: 500200, Y: 5.004e+06

Cleaning down active device memory tracking layouts...
Harness run execution passed successfully.
