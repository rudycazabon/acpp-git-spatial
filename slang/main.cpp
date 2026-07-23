// main.cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <memory>
#include <cstring>

#include <slang.h>
#include <slang-com-ptr.h>
#include <slang-gfx.h>

using namespace Slang;
using namespace gfx;

struct UtmInput {
    double easting; double northing;
    int zone; int isNorthernHemisphere; int padding;
};

struct OutputCoords { double x; double y; };

int main() {
    const size_t datasetSize = 1000000;
    std::cout << "Initializing Slang 1.4.350 Explicit Compute Engine Pipeline...\n";

    gfx::IDevice::Desc deviceDesc = {};
    deviceDesc.deviceType = gfx::DeviceType::Vulkan; 
    
    Slang::ComPtr<gfx::IDevice> device;
    if (SLANG_FAILED(gfx::gfxCreateDevice(&deviceDesc, device.writeRef()))) {
        std::cerr << "Fatal error: Failed to initialize Vulkan context.\n";
        return -1;
    }

    slang::ISession* slangSession = device->getSlangSession();
    slang::IModule* slangModule = nullptr;
    Slang::ComPtr<slang::IBlob> diagnosticsBlob;
    slangModule = slangSession->loadModule("spatial_transform", diagnosticsBlob.writeRef());
    if (!slangModule) {
        std::cerr << "Compilation error:\n" << (const char*)diagnosticsBlob->getBufferPointer() << "\n";
        return -1;
    }

    Slang::ComPtr<slang::IEntryPoint> entryPoint;
    slangModule->findEntryPointByName("computeMain", entryPoint.writeRef());

    slang::IComponentType* components[] = { slangModule, entryPoint.get() };
    Slang::ComPtr<slang::IComponentType> composedProgram;
    slangSession->createCompositeComponentType(components, 2, composedProgram.writeRef());

    gfx::IShaderProgram::Desc programDesc = {};
    programDesc.slangGlobalScope = composedProgram.get();
    
    Slang::ComPtr<gfx::IShaderProgram> shaderProgram;
    if (SLANG_FAILED(device->createProgram(programDesc, shaderProgram.writeRef()))) {
        std::cerr << "Fatal error: Failed to link shader program compilation state.\n";
        return -1;
    }

    gfx::ComputePipelineStateDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram.get();
    
    ComPtr<gfx::IPipelineState> pipelineState;
    if (SLANG_FAILED(device->createComputePipelineState(pipelineDesc, pipelineState.writeRef()))) {
        std::cerr << "Fatal error: Failed to compile pipeline layout.\n";
        return -1;
    }

    size_t inBufferSize = datasetSize * sizeof(UtmInput);
    size_t outBufferSize = datasetSize * sizeof(OutputCoords);

    gfx::IBufferResource::Desc inBufferDesc = {};
    inBufferDesc.sizeInBytes = inBufferSize;
    inBufferDesc.format = gfx::Format::Unknown;
    inBufferDesc.elementSize = sizeof(UtmInput);
    inBufferDesc.memoryType = gfx::MemoryType::DeviceLocal;
    inBufferDesc.defaultState = gfx::ResourceState::ShaderResource;

    gfx::IBufferResource::Desc outBufferDesc = {};
    outBufferDesc.sizeInBytes = outBufferSize;
    outBufferDesc.format = gfx::Format::Unknown;
    outBufferDesc.elementSize = sizeof(OutputCoords);
    outBufferDesc.memoryType = gfx::MemoryType::DeviceLocal;
    outBufferDesc.defaultState = gfx::ResourceState::UnorderedAccess;

    ComPtr<gfx::IBufferResource> inputBuffer  = device->createBufferResource(inBufferDesc);
    ComPtr<gfx::IBufferResource> outputBuffer = device->createBufferResource(outBufferDesc);

    std::vector<UtmInput> hostInputs(datasetSize);
    for (size_t i = 0; i < datasetSize; ++i) {
        hostInputs[i] = { 
            500000.0 + (i * 0.01), 
            5000000.0 + (i * 0.01), 
            32, 
            1, 
            0 
        };
    }

    void* mappedPtr = nullptr;
    if (SLANG_SUCCEEDED(inputBuffer->map(nullptr, &mappedPtr))) {
        std::memcpy(mappedPtr, hostInputs.data(), inBufferSize);
        inputBuffer->unmap(nullptr);
    } else {
        std::cerr << "Fatal error: Input memory staging mapping failed.\n";
        return -1;
    }

    ComPtr<gfx::ITransientResourceHeap> resourceHeap;
    gfx::ITransientResourceHeap::Desc heapDesc = {};
    heapDesc.constantBufferSize = 16 * 1024;
    device->createTransientResourceHeap(heapDesc, resourceHeap.writeRef());

    slang::ProgramLayout* programLayout = composedProgram->getLayout();
    slang::TypeReflection* typeReflection = programLayout->getGlobalParamsVarLayout()->getType();
    ComPtr<gfx::IShaderObject> rootShaderObject;
    device->createShaderObject(typeReflection, gfx::ShaderObjectContainerType::None, rootShaderObject.writeRef());

    gfx::ShaderOffset inputOffset = { 0, 0 };
    gfx::ShaderOffset outputOffset = { 1, 0 };
    
    gfx::IResourceView::Desc inViewDesc = {};
    inViewDesc.type = gfx::IResourceView::Type::ShaderResource;
    ComPtr<gfx::IResourceView> inputView = device->createBufferView(inputBuffer.get(), nullptr, inViewDesc);
    
    gfx::IResourceView::Desc outViewDesc = {};
    outViewDesc.type = gfx::IResourceView::Type::UnorderedAccess;
    ComPtr<gfx::IResourceView> outputView = device->createBufferView(outputBuffer.get(), nullptr, outViewDesc);

    rootShaderObject->setResource(inputOffset, inputView.get());
    rootShaderObject->setResource(outputOffset, outputView.get());

    ComPtr<gfx::ICommandBuffer> commandBuffer = resourceHeap->createCommandBuffer();
    gfx::IComputeCommandEncoder* encoder = commandBuffer->encodeComputeCommands();

    encoder->bindPipeline(pipelineState.get());
    
    // CRITICAL FIX: Explicitly bind the descriptor layout tree state to the encoder pass 
    // This tells Vulkan to pass the rootShaderObject buffers down to the executing pipeline.
    encoder->bindPipelineWithRootObject(pipelineState.get(), rootShaderObject.get());

    encoder->dispatchCompute(static_cast<int>((datasetSize + 255) / 256), 1, 1);
    encoder->endEncoding();

    std::cout << "Dispatching Slang compute kernel to GPU...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    gfx::ICommandQueue::Desc queueDesc = {};
    queueDesc.type = gfx::ICommandQueue::QueueType::Graphics;
    ComPtr<gfx::ICommandQueue> queue = device->createCommandQueue(queueDesc);
    queue->executeCommandBuffer(commandBuffer.get());
    queue->waitOnHost();

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "GPU Execution Completed inside: " 
              << std::chrono::duration<double, std::milli>(end - start).count() << " ms.\n";

    std::vector<OutputCoords> hostOutputs(datasetSize);
    ComPtr<ISlangBlob> outBlob;
    if (SLANG_SUCCEEDED(device->readBufferResource(outputBuffer.get(), 0, outBufferSize, outBlob.writeRef()))) {
        std::memcpy(hostOutputs.data(), outBlob->getBufferPointer(), outBufferSize);
        
        // Fixed pointer logging error index
        std::cout << "Sample Vertex 0 Ground Truth -> X: " << hostOutputs[0].x 
                  << ", Y: " << hostOutputs[0].y << "\n";
    } else {
        std::cerr << "Error: Staging read-back failed.\n";
    }

    return 0;
}
