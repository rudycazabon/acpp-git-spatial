#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// ============================================================================
// DATA STRUCT DEFINITIONS (Ensuring clean hardware-aligned memory bounds)
// ============================================================================

struct UtmInputDouble {
    double easting;
    double northing;
    int zone;
    int isNorthernHemisphere;
    int padding; 
};

struct LocalUtmInputFloat {
    float deltaEasting;
    float deltaNorthing;
};

struct OutputCoords {
    double x;
    double y;
};

// Constant parameters for global transformation
struct TransformConstants {
    const double wgs84_a = 6378137.0;
    const double wgs84_f = 1.0 / 298.257223563;
    const double grs80_a = 6378137.0;
    const double grs80_f = 1.0 / 298.257222101;
    // 7 Helmert Parameters (WGS84 -> EVRF2007)
    const double tx = 0.054;   const double ty = 0.051;   const double tz = -0.114;
    const double rx = 2.44e-8; const double ry = 2.41e-8; const double rz = -4.70e-8;
    const double s  = 1.000000011;
};

// ============================================================================
// SYCL IMPL 1: NATIVE FP64 ACCELERATION PIPELINE (Scoping Fixed)
// ============================================================================
void runNativeFP64Pipeline(sycl::queue& q, const std::vector<UtmInputDouble>& hostInputs, size_t numElements) {
    std::cout << "\n--- Starting Native FP64 Execution Pass ---\n";

    std::vector<OutputCoords> hostOutputs(numElements);
    auto start = std::chrono::high_resolution_clock::now();

    // EXPLICIT SCOPING BLOCK: Unlocks automatic synchronization upon destruction
    {
        sycl::buffer<UtmInputDouble, 1> inBuffer(hostInputs.data(), sycl::range<1>(numElements));
        sycl::buffer<OutputCoords, 1> outBuffer(hostOutputs.data(), sycl::range<1>(numElements));

        q.submit([&](sycl::handler& cgh) {
            auto inAcc  = inBuffer.get_access<sycl::access::mode::read>(cgh);
            auto outAcc = outBuffer.get_access<sycl::access::mode::write>(cgh);
            TransformConstants c;

            cgh.parallel_for<class NativeHelmertFP64>(sycl::range<1>(numElements), [=](sycl::id<1> idx) {
                size_t id = idx;
                UtmInputDouble utm = inAcc[id];

                // 1. UTM -> WGS84 Geodetic conversion
                double lon0 = ((double(utm.zone) - 1.0) * 6.0 - 180.0 + 3.0) * (3.1415926535897932 / 180.0);
                double x_norm = (utm.easting - 500000.0) / 0.9996;
                double y_norm = (utm.northing - (utm.isNorthernHemisphere ? 0.0 : 10000000.0)) / 0.9996;
                double lat = y_norm / c.wgs84_a;
                double lon = lon0 + (x_norm / c.wgs84_a);
                lat += (2.0 * c.wgs84_f - c.wgs84_f * c.wgs84_f) * sycl::sin(2.0 * lat);

                // 2. Geodetic -> ECEF -> Matrix Shift -> Geodetic (Helmert)
                double sinLat = sycl::sin(lat); double cosLat = sycl::cos(lat);
                double e2_src = 2.0 * c.wgs84_f - c.wgs84_f * c.wgs84_f;
                double N = c.wgs84_a / sycl::sqrt(1.0 - e2_src * sinLat * sinLat);
                
                double wgsEcefX = N * cosLat * sycl::cos(lon);
                double wgsEcefY = N * cosLat * sycl::sin(lon);
                double wgsEcefZ = N * (1.0 - e2_src) * sinLat;

                double evrfEcefX = c.tx + c.s * (wgsEcefX - c.rz * wgsEcefY + c.ry * wgsEcefZ);
                double evrfEcefY = c.ty + c.s * (c.rz * wgsEcefX + wgsEcefY - c.rx * wgsEcefZ);
                double evrfEcefZ = c.tz + c.s * (-c.ry * wgsEcefX + c.rx * wgsEcefY + wgsEcefZ);

                double e2_dst = 2.0 * c.grs80_f - c.grs80_f * c.grs80_f;
                double p = sycl::sqrt(evrfEcefX * evrfEcefX + evrfEcefY * evrfEcefY);
                double outLon = sycl::atan2(evrfEcefY, evrfEcefX);
                double outLat = sycl::atan2(evrfEcefZ, p * (1.0 - e2_dst));
                
                for(int i = 0; i < 3; i++) {
                    double sinPt = sycl::sin(outLat);
                    double N_dst = c.grs80_a / sycl::sqrt(1.0 - e2_dst * sinPt * sinPt);
                    outLat = sycl::atan2(evrfEcefZ + e2_dst * N_dst * sinPt, p);
                }

                // 3. Project to final Destination Ellipsoidal Mercator
                double e_m = sycl::sqrt(e2_dst);
                outAcc[id].x = c.grs80_a * outLon;
                outAcc[id].y = c.grs80_a * sycl::log(sycl::tan((3.1415926535897932 / 4.0) + (outLat / 2.0)) * 
                               sycl::pow((1.0 - e_m * sycl::sin(outLat)) / (1.0 + e_m * sycl::sin(outLat)), e_m / 2.0));
            });
        });
    } // <-- outBuffer and inBuffer are destroyed here, flushing GPU data back to hostOutputs cleanly

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "FP64 Pipeline Finished in " << duration.count() << " ms.\n";
    // Fixed: Added explicit vector indexing [0]
    std::cout << "Sample Output 0 -> X: " << hostOutputs[0].x << ", Y: " << hostOutputs[0].y << "\n";
}

// ============================================================================
// SYCL IMPL 2: CAMERA-RELATIVE ANCHOR PIPELINE (Scoping Fixed)
// ============================================================================
void runCameraRelativeFP32Pipeline(sycl::queue& q, const std::vector<LocalUtmInputFloat>& hostInputs, size_t numElements) {
    std::cout << "\n--- Starting Camera-Relative FP32 Execution Pass ---\n";

    const float anchorMercatorX = 500200.0f;
    const float anchorMercatorY = 5004000.0f;
    const float j_dX_dE = 1.00012f;  const float j_dX_dN = 0.00001f;
    const float j_dY_dE = -0.00002f; const float j_dY_dN = 1.00034f;

    std::vector<OutputCoords> hostOutputs(numElements);
    auto start = std::chrono::high_resolution_clock::now();

    // EXPLICIT SCOPING BLOCK
    {
        sycl::buffer<LocalUtmInputFloat, 1> inBuffer(hostInputs.data(), sycl::range<1>(numElements));
        sycl::buffer<OutputCoords, 1> outBuffer(hostOutputs.data(), sycl::range<1>(numElements));

        q.submit([&](sycl::handler& cgh) {
            auto inAcc  = inBuffer.get_access<sycl::access::mode::read>(cgh);
            auto outAcc = outBuffer.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<class RelativeAnchorFP32>(sycl::range<1>(numElements), [=](sycl::id<1> idx) {
                size_t id = idx;
                LocalUtmInputFloat target = inAcc[id];

                float localMercatorX = (target.deltaEasting * j_dX_dE) + (target.deltaNorthing * j_dX_dN);
                float localMercatorY = (target.deltaEasting * j_dY_dE) + (target.deltaNorthing * j_dY_dN);

                outAcc[id].x = double(anchorMercatorX + localMercatorX);
                outAcc[id].y = double(anchorMercatorY + localMercatorY);
            });
        });
    } // <-- GPU unmaps here safely

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "FP32 Relative Pipeline Finished in " << duration.count() << " ms.\n";
    // Fixed: Added explicit vector indexing [0]
    std::cout << "Sample Output 0 -> X: " << hostOutputs[0].x << ", Y: " << hostOutputs[0].y << "\n";
}


// ============================================================================
// MAIN EXECUTION TRACKER HARNESS
// ============================================================================
int main() {
    const size_t datasetSize = 1000000; // Benchmark on 1 million geospatial coordinates
    std::cout << "Allocating test payloads for " << datasetSize << " structural entities...\n";

    // Generate simulated test datasets
    std::vector<UtmInputDouble> inputFp64(datasetSize);
    std::vector<LocalUtmInputFloat> inputFp32(datasetSize);

    for (size_t i = 0; i < datasetSize; ++i) {
        inputFp64[i] = { 500000.0 + (i * 0.01), 5000000.0 + (i * 0.01), 32, 1, 0 };
        inputFp32[i] = { static_cast<float>(i * 0.01f), static_cast<float>(i * 0.01f) };
    }

    // Initialize SYCL execution queues using AdaptiveCPP selectors
    try {
        // Attempt to request an accelerator device (GPU)
        sycl::queue gpuQueue(sycl::gpu_selector_v);
        std::cout << "Target Device (GPU Backend): " << gpuQueue.get_device().get_info<sycl::info::device::name>() << "\n";
        
        runNativeFP64Pipeline(gpuQueue, inputFp64, datasetSize);
        runCameraRelativeFP32Pipeline(gpuQueue, inputFp32, datasetSize);
        
    } catch (const sycl::exception& e) {
        std::cerr << "GPU Acceleration context unavailable: " << e.what() << "\nFalling back to CPU execution processing topology...\n";
        
        // Graceful fallback to CPU thread groups
        sycl::queue cpuQueue(sycl::cpu_selector_v);
        std::cout << "Target Device (CPU Backend): " << cpuQueue.get_device().get_info<sycl::info::device::name>() << "\n";
        
        runNativeFP64Pipeline(cpuQueue, inputFp64, datasetSize);
        runCameraRelativeFP32Pipeline(cpuQueue, inputFp32, datasetSize);
    }

    return 0;
}
