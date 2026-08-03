// double_minimal.slang
// Structurally identical to the confirmed-working float minimal example --
// single scalar field, no mixed types, no interleaving -- to isolate whether
// this is a double-vs-float problem or a struct-layout problem.

public struct InputData {
    public double value;
};

public struct OutputData {
    public double result;
};

public StructuredBuffer<InputData> inputBuffer;
public RWStructuredBuffer<OutputData> outputBuffer;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 threadId : SV_DispatchThreadID) {
    uint id = threadId.x;
    InputData data = inputBuffer[id];
    // Explicit double literal (note the L suffix) to rule out the
    // implicit-float-literal-then-widen path the compiler warned about
    // in the real shader (unsuffixed literals default to float in
    // HLSL/Slang and get implicitly widened -- ruling that out here).
    outputBuffer[id].result = data.value * 2.0L;
}
