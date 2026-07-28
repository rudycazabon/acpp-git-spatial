// spatial_transform.slang

public struct UtmInput {
    public double easting;
    public double northing;
    public int zone;
    public int isNorthernHemisphere;
    public int padding;
};

public struct OutputCoords {
    public double x;
    public double y;
};

struct TransformConstants {
    double wgs84_a = 6378137.0;
    double wgs84_f = 1.0 / 298.257223563;
    double grs80_a = 6378137.0;
    double grs80_f = 1.0 / 298.257222101;
    double tx = 0.054;   double ty = 0.051; double tz = -0.114;
    double rx = 2.44e-8; double ry = 2.41e-8; double rz = -4.70e-8;
    double s  = 1.00000011;
};

// Parameter Blocks simplify binding layout generation 
public ParameterBlock<StructuredBuffer<UtmInput>> inputBuffer;
public ParameterBlock<RWStructuredBuffer<OutputCoords>> outputBuffer;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 threadId : SV_DispatchThreadID) {
    uint id = threadId.x;
    TransformConstants c;

    UtmInput utm = inputBuffer[id];

    // UTM -> WGS84 Geodetic
    double lon0 = ((double(utm.zone) - 1.0) * 6.0 - 180.0 + 3.0) * (3.1415926535897932 / 180.0);
    double x_norm = (utm.easting - 500000.0) / 0.9996;
    double y_norm = (utm.northing - (utm.isNorthernHemisphere ? 0.0 : 10000000.0)) / 0.9996;
    double lat = y_norm / c.wgs84_a;
    double lon = lon0 + (x_norm / c.wgs84_a);
    lat += (2.0 * c.wgs84_f - c.wgs84_f * c.wgs84_f) * sin(2.0 * lat);

    // Geodetic -> ECEF -> Helmert Shift -> Geodetic
    double sinLat = sin(lat); double cosLat = cos(lat);
    double e2_src = 2.0 * c.wgs84_f - c.wgs84_f * c.wgs84_f;
    double N = c.wgs84_a / sqrt(1.0 - e2_src * sinLat * sinLat);
    
    double wgsEcefX = N * cosLat * cos(lon);
    double wgsEcefY = N * cosLat * sin(lon);
    double wgsEcefZ = N * (1.0 - e2_src) * sinLat;

    double evrfEcefX = c.tx + c.s * (wgsEcefX - c.rz * wgsEcefY + c.ry * wgsEcefZ);
    double evrfEcefY = c.ty + c.s * (c.rz * wgsEcefX + wgsEcefY - c.rx * wgsEcefZ);
    double evrfEcefZ = c.tz + c.s * (-c.ry * wgsEcefX + c.rx * wgsEcefY + wgsEcefZ);

    double e2_dst = 2.0 * c.grs80_f - c.grs80_f * c.grs80_f;
    double p = sqrt(evrfEcefX * evrfEcefX + evrfEcefY * evrfEcefY);
    double outLon = atan2(evrfEcefY, evrfEcefX);
    double outLat = atan2(evrfEcefZ, p * (1.0 - e2_dst));
    
    for(int i = 0; i < 3; i++) {
        double sinPt = sin(outLat);
        double N_dst = c.grs80_a / sqrt(1.0 - e2_dst * sinPt * sinPt);
        outLat = atan2(evrfEcefZ + e2_dst * N_dst * sinPt, p);
    }

    // Project to Final Ellipsoidal Mercator
    double e_m = sqrt(e2_dst);
    outputBuffer[id].x = c.grs80_a * outLon;
    outputBuffer[id].y = c.grs80_a * log(tan((3.1415926535897932 / 4.0) + (outLat / 2.0)) * 
                   pow((1.0 - e_m * sin(outLat)) / (1.0 + e_m * sin(outLat)), e_m / 2.0));
}
