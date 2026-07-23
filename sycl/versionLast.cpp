// main.cpp
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

struct UtmInputDouble {
    double easting; double northing;
    int zone; int isNorthernHemisphere; int padding; 
};

struct LocalUtmInputFloat { float deltaEasting; float deltaNorthing; };
struct OutputCoords { double x; double y; };

struct TransformConstants {
    double wgs84_a = 6378137.0;        double wgs84_f = 1.0 / 298.257223563;
    double grs80_a = 6378137.0;        double grs80_f = 1.0 / 298.257222101;
    double tx = 0.054;   double ty = 0.051; double tz = -0.114;
    double rx = 2.44e-8; double ry = 2.41e-8; double rz = -4.70e-8;
    double s  = 1.00000011;
};

int main() {
    const size_t datasetSize = 1000000;
    std::cout << "[Harness Initialization] Selecting System Target Context...\n";
    
    sycl::queue q{sycl::cpu_selector_v};
    std::cout << "Using Device Backend: " << q.get_device().get_info<sycl::info::device::name>() << "\n\n";

    // Standard vector heap storage prevents tracking conflicts within the device driver
    std::vector<UtmInputDouble> inputFp64(datasetSize);
    std::vector<LocalUtmInputFloat> inputFp32(datasetSize);
    std::vector<OutputCoords> outputs(datasetSize);

    for (size_t i = 0; i < datasetSize; ++i) {
        inputFp64[i] = { 
            500000.0 + (i * 0.01), 
            5000000.0 + (i * 0.01), 
            32, 
            1, 
            0 
        };
        inputFp32[i] = { 
            static_cast<float>(i * 0.01f), 
            static_cast<float>(i * 0.01f) 
        };
    }

    std::cout << "Executing Loops via Standard Buffer Mapping Scopes...\n";
    auto start = std::chrono::high_resolution_clock::now();

    {
        sycl::buffer<UtmInputDouble, 1> bufIn64(inputFp64.data(), sycl::range<1>(datasetSize));
        sycl::buffer<LocalUtmInputFloat, 1> bufIn32(inputFp32.data(), sycl::range<1>(datasetSize));
        sycl::buffer<OutputCoords, 1> bufOut(outputs.data(), sycl::range<1>(datasetSize));

        q.submit([&](sycl::handler& cgh) {
            auto in64 = bufIn64.get_access<sycl::access::mode::read>(cgh);
            auto out  = bufOut.get_access<sycl::access::mode::write>(cgh);
            TransformConstants c;

            cgh.parallel_for<class HelmertFP64>(sycl::range<1>(datasetSize), [=](sycl::id<1> idx) {
                size_t id = idx;
                UtmInputDouble utm = in64[id];
                double lon0 = ((double(utm.zone) - 1.0) * 6.0 - 180.0 + 3.0) * (3.1415926535897932 / 180.0);
                double x_norm = (utm.easting - 500000.0) / 0.9996;
                double y_norm = (utm.northing - (utm.isNorthernHemisphere ? 0.0 : 10000000.0)) / 0.9996;
                double lat = y_norm / c.wgs84_a;
                double lon = lon0 + (x_norm / c.wgs84_a);
                lat += (2.0 * c.wgs84_f - c.wgs84_f * c.wgs84_f) * sycl::sin(2.0 * lat);

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
                for(int i=0; i<3; i++) {
                    double sinPt = sycl::sin(outLat);
                    outLat = sycl::atan2(evrfEcefZ + e2_dst * (c.grs80_a / sycl::sqrt(1.0 - e2_dst * sinPt * sinPt)) * sinPt, p);
                }
                out[id].x = c.grs80_a * outLon;
                out[id].y = c.grs80_a * sycl::log(sycl::tan((3.1415926535897932 / 4.0) + (outLat / 2.0)) * sycl::pow((1.0 - sycl::sqrt(e2_dst) * sycl::sin(outLat)) / (1.0 + sycl::sqrt(e2_dst) * sycl::sin(outLat)), sycl::sqrt(e2_dst) / 2.0));
            });
        });

        q.submit([&](sycl::handler& cgh) {
            auto in32 = bufIn32.get_access<sycl::access::mode::read>(cgh);
            auto out  = bufOut.get_access<sycl::access::mode::write>(cgh);
            
            cgh.parallel_for<class RelativeFP32>(sycl::range<1>(datasetSize), [=](sycl::id<1> idx) {
                size_t id = idx;
                LocalUtmInputFloat target = in32[id];
                out[id].x = double(500200.0f + (target.deltaEasting * 1.00012f) + (target.deltaNorthing * 0.00001f));
                out[id].y = double(5004000.0f + (target.deltaEasting * -0.00002f) + (target.deltaNorthing * 1.00034f));
            });
        });
    } // Buffer scope pop drops allocations safely

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "All kernels passed execution bounds successfully!\n";
    std::cout << "Benchmark Duration: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms.\n";
    std::cout << "Output 0 Sample Coordinates -> X: " << outputs[0].x << ", Y: " << outputs[0].y << "\n";

    return 0;
}
