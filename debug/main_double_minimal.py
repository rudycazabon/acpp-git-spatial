import slangpy as spy
import numpy as np
import pathlib

device = spy.create_device(
    include_paths=[pathlib.Path(__file__).parent.absolute()],
    enable_debug_layers=True,
)
program = device.load_program(module_name="double_minimal.slang", entry_point_names=["computeMain"])
kernel = device.create_compute_kernel(program=program)

input_dtype = np.dtype({'names': ['value'], 'formats': [np.float64]}, align=True)

n = 8
input_data = np.zeros(n, dtype=input_dtype)
input_data['value'] = np.arange(1, n + 1, dtype=np.float64)

print("numpy itemsize:", input_dtype.itemsize)
print(kernel.reflection.inputBuffer)

input_buffer = device.create_buffer(
    element_count=n,
    resource_type_layout=kernel.reflection.inputBuffer,
    usage=spy.BufferUsage.shader_resource,
    data=input_data.view(np.uint8).reshape(-1),
)

output_buffer = device.create_buffer(
    element_count=n,
    resource_type_layout=kernel.reflection.outputBuffer,
    usage=spy.BufferUsage.unordered_access,
)

kernel.dispatch(
    thread_count=[n, 1, 1],
    inputBuffer=input_buffer,
    outputBuffer=output_buffer,
)

output_dtype = np.dtype({'names': ['result'], 'formats': [np.float64]}, align=True)
result = output_buffer.to_numpy().view(output_dtype)
print("Expected:", input_data['value'] * 2.0)
print("Got:     ", result['result'])
