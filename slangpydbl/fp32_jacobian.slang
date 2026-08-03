// fp32_jacobian.slang
// GPU-side: purely float. All fp64 work (computing anchorMercatorX/Y and the
// Jacobian partial derivatives) happens on the host in Python/numpy using
// float64, and only the resulting float32 scalars are uploaded here.

public struct LocalUtmInput {
    public float deltaEasting;   // offset from anchor, in meters
    public float deltaNorthing;  // offset from anchor, in meters
};

public struct RelativeOutput {
    public float x;  // Mercator X, meters
    public float y;  // Mercator Y, meters
};

public StructuredBuffer<LocalUtmInput> inputBuffer;
public RWStructuredBuffer<RelativeOutput> outputBuffer;

// Plain global uniforms -- bound the same way inputBuffer/outputBuffer are,
// via kwargs on kernel.dispatch(), which is the pattern already confirmed
// working earlier in this session.
public float anchorMercatorX;
public float anchorMercatorY;
public float j_dX_dE;
public float j_dX_dN;
public float j_dY_dE;
public float j_dY_dN;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 threadId : SV_DispatchThreadID) {
    uint id = threadId.x;
    LocalUtmInput target = inputBuffer[id];

    // Pure linear approximation -- no trig, no double, anywhere.
    float localX = target.deltaEasting * j_dX_dE + target.deltaNorthing * j_dX_dN;
    float localY = target.deltaEasting * j_dY_dE + target.deltaNorthing * j_dY_dN;

    outputBuffer[id].x = anchorMercatorX + localX;
    outputBuffer[id].y = anchorMercatorY + localY;
}
