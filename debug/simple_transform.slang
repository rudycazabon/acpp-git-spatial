// simple_transform.slang

public struct InputData {
    public float value;
    public int id;
};

public struct OutputData {
    public float result;
};

public StructuredBuffer<InputData> inputBuffer;
public RWStructuredBuffer<OutputData> outputBuffer;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 threadId: SV_DispatchThreadID) {
    uint id = threadId.x;
    InputData data = inputBuffer[id];
    outputBuffer[id].result = data.value * 2.0;
}