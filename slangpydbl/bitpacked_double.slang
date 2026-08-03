// bitpacked_double.slang
//
// NOTE: asdouble()/asuint(double, out uint, out uint) are standard HLSL
// intrinsics. Slang is HLSL-compatible for these, but this specific pairing
// hasn't been build-tested in this session yet -- if it fails to compile,
// that's the first thing to check (Slang's exact intrinsic signature/name
// may differ slightly; check with `slangc -entry computeMain -profile sm_6_0`
// or similar before assuming the math itself is wrong).

public struct UtmInputPacked {
    public uint easting_lo;
    public uint easting_hi;
    public uint northing_lo;
    public uint northing_hi;
    public int zone;
    public int isNorthernHemisphere;
};

public struct OutputCoordsPacked {
    public uint x_lo;
    public uint x_hi;
    public uint y_lo;
    public uint y_hi;
};

public StructuredBuffer<UtmInputPacked> inputBuffer;
public RWStructuredBuffer<OutputCoordsPacked> outputBuffer;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 threadId : SV_DispatchThreadID) {
    uint id = threadId.x;
    UtmInputPacked packed = inputBuffer[id];

    // Reassemble doubles from their uint32 halves.
    double easting  = asdouble(packed.easting_lo, packed.easting_hi);
    double northing = asdouble(packed.northing_lo, packed.northing_hi);

    // --- Real double-precision math goes here -----------------------------
    // Replace this placeholder with the actual UTM -> Helmert -> Mercator
    // pipeline; kept trivial here to isolate the packing mechanism itself.
    double resultX = easting * 1.0002;
    double resultY = northing * 1.0005;
    // ------------------------------------------------------------------------

    // Split results back into uint32 halves for storage.
    uint xlo, xhi, ylo, yhi;
    asuint(resultX, xlo, xhi);
    asuint(resultY, ylo, yhi);

    outputBuffer[id].x_lo = xlo;
    outputBuffer[id].x_hi = xhi;
    outputBuffer[id].y_lo = ylo;
    outputBuffer[id].y_hi = yhi;
}
