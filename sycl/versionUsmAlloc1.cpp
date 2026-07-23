#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// ============================================================================
// DATA STRUCT DEFINITIONS (Clean Alignment)
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

struct TransformConstants {
    const double wgs84_a = 6378137.0;
    const double wgs84_f = 1.0 / 298.257223563;
    const double grs80_a = 6378137.0;
    const double grs80_f = 1.0 / 298.257222101;
    const double tx = 0.054;   const double ty = 0.051;   const double tz = -0.114;
    const double rx = 2.44e-8; const double ry = 2.41e-8; const double rz = -4.70e-8;
    const double s  = 1.00000011;
};

// ============================================================================
// SYCL IMPL 1: NATIVE USM FP64 PIPELINE
// ============================================================================
void runNativeFP64Pipeline(sycl::queue& q, const UtmInputDouble* d_inputs, OutputCoords* d_outputs, size_t numElements) {
    std::cout << "\n--- Starting Native USM FP64 Execution Pass ---\n";

    auto start = std::chrono::high_resolution_clock::now();

    q.submit([&](sycl::handler& cgh) {
        TransformConstants c;

        cgh.parallel_for<class NativeHelmertFP64USM>(sycl::range<1>(numElements), [=](sycl::id<1> idx) {
            size_t id = idx;
            UtmInputDouble utm = d_inputs[id];

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
            d_outputs[id].x = c.grs80_a * outLon;
            d_outputs[id].y = c.grs80_a * sycl::log(sycl::tan((3.1415926535897932 / 4.0) + (outLat / 2.0)) * 
                           sycl::pow((1.0 - e_m * sycl::sin(outLat)) / (1.0 + e_m * sycl::sin(outLat)), e_m / 2.0));
        });
    });

    q.wait(); // Blocks until kernels execute cleanly
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "USM FP64 Pipeline Finished in " << duration.count() << " ms.\n";
    std::cout << "Sample Output 0 -> X: " << d_outputs[0].x << ", Y: " << d_outputs[0].y << "\n";
}

// ============================================================================
// SYCL IMPL 2: CAMERA-RELATIVE USM FP32 PIPELINE
// ============================================================================
void runCameraRelativeFP32Pipeline(sycl::queue& q, const LocalUtmInputFloat* d_inputs, OutputCoords* d_outputs, size_t numElements) {
    std::cout << "\n--- Starting Camera-Relative USM FP32 Execution Pass ---\n";

    const float anchorMercatorX = 500200.0f;
    const float anchorMercatorY = 5004000.0f;
    const float j_dX_dE = 1.00012f;  const float j_dX_dN = 0.00001f;
    const float j_dY_dE = -0.00002f; const float j_dY_dN = 1.00034f;

    auto start = std::chrono::high_resolution_clock::now();

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<class RelativeAnchorFP32USM>(sycl::range<1>(numElements), [=](sycl::id<1> idx) {
            size_t id = idx;
            LocalUtmInputFloat target = d_inputs[id];

            float localMercatorX = (target.deltaEasting * j_dX_dE) + (target.deltaNorthing * j_dX_dN);
            float localMercatorY = (target.deltaEasting * j_dY_dE) + (target.deltaNorthing * j_dY_dN);

            d_outputs[id].x = double(anchorMercatorX + localMercatorX);
            d_outputs[id].y = double(anchorMercatorY + localMercatorY);
        });
    });

    q.wait();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "USM FP32 Relative Pipeline Finished in " << duration.count() << " ms.\n";
    std::cout << "Sample Output 0 -> X: " << d_outputs[0].x << ", Y: " << d_outputs[0].y << "\n";
}

// ============================================================================
// MAIN EXECUTION ENGINE
// ============================================================================
int main() {
    const size_t datasetSize = 1000000;
    std::cout << "Initializing SYCL Queue Context and Shared USM Allocations...\n";

    sycl::queue q;
    try {
        q = sycl::queue(sycl::gpu_selector_v);
    } catch (const sycl::exception& e) {
        std::cout << "GPU context requested failed, initializing fallback Host CPU selector context...\n";
        q = sycl::queue(sycl::cpu_selector_v);
    }
    std::cout << "Running on Active Hardware Instance: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    // 1. Allocate Unified Shared Memory Pools
    UtmInputDouble*     inputFp64 = sycl::malloc_shared<UtmInputDouble>(datasetSize, q);
    LocalUtmInputFloat* inputFp32 = sycl::malloc_shared<LocalUtmInputFloat>(datasetSize, q);
    OutputCoords*       outputs   = sycl::malloc_shared<OutputCoords>(datasetSize, q);

    if (!inputFp64 || !inputFp32 || !outputs) {
        std::cerr << "Fatal error: Shared Unified Memory allocation failed.\n";
        return -1;
    }

    // 2. Initialize values inside the shared page boundaries on the host thread
    for (size_t i = 0; i < datasetSize; ++i) {
        inputFp64[i] = { 500000.0 + (i * 0.01), 5000000.0 + (i * 0.01), 32, 1, 0 };
        inputFp32[i] = { static_cast<float>(i * 0.01f), static_cast<float>(i * 0.01f) };
        outputs[i]   = { 0.0, 0.0 };
    }

    // 3. Process the workloads
    runNativeFP64Pipeline(q, inputFp64, outputs, datasetSize);
    runCameraRelativeFP32Pipeline(q, inputFp32, outputs, datasetSize);

    // 4. Safely free up the device driver spaces without breaking standard host pointers
    std::cout << "\nCleaning down active device memory tracking layouts...\n";
    sycl::free(inputFp64, q);
    sycl::free(inputFp32, q);
    sycl::free(outputs, q);

    std::cout << "Harness run execution passed successfully.\n";
    return 0;
}
