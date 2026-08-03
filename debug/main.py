import slangpy as spy
import numpy as np
import pathlib

device = spy.create_device(include_paths=[pathlib.Path(__file__).parent.absolute()])
program = device.load_program(module_name="simple_transform.slang", entry_point_names=["computeMain"])
kernel = device.create_compute_kernel(program=program)

# Mirror the Slang struct field-for-field, in the same order, with align=True
input_dtype = np.dtype({
    'names':   ['value', 'id'],
    'formats': [np.float32, np.int32],
}, align=True)

n = 8
input_data = np.zeros(n, dtype=input_dtype)
input_data['value'] = np.arange(n, dtype=np.float32)
input_data['id']    = np.arange(n, dtype=np.int32)

# Sanity check before creating the buffer: these two numbers MUST match
print("numpy itemsize:", input_dtype.itemsize)
print(kernel.reflection.inputBuffer)  # look for element_type_layout.stride in the printout

input_buffer = device.create_buffer(
    element_count=n,
    resource_type_layout=kernel.reflection.inputBuffer,
    usage=spy.BufferUsage.shader_resource,
    data=input_data.view(np.uint8).reshape(-1),   # structured dtype -> flat bytes
)

output_buffer = device.create_buffer(
    element_count=n,
    resource_type_layout=kernel.reflection.outputBuffer,
    usage=spy.BufferUsage.unordered_access,
)