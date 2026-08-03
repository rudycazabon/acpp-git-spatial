import slangpy as spy
import numpy as np
import pathlib

device = spy.create_device(include_paths=[
        pathlib.Path(__file__).parent.absolute(),
])


program = device.load_program(
    module_name="spatial_transform.slang",
    entry_point_names=["computeMain"],
)
kernel = device.create_compute_kernel(
    program=program
)

# UtmInput: double, double, int, int, int -> padded to 32 bytes (align 8)
utm_dtype = np.dtype({
    'names':   ['easting', 'northing', 'zone', 'isNorthernHemisphere', 'padding'],
    'formats': [np.float64, np.float64, np.int32, np.int32, np.int32],
}, align=True)

# OutputCoords: double, double -> 16 bytes, no padding needed
coords_dtype = np.dtype({
    'names':   ['x', 'y'],
    'formats': [np.float64, np.float64],
}, align=True)

print(utm_dtype.itemsize)     # sanity check -> should be 32
print(coords_dtype.itemsize)  # -> should be 16

print(kernel.reflection.inputBuffer)   # inspect what slangpy sees
print(kernel.reflection.outputBuffer)

n = 1024

utm_data = np.zeros(n, dtype=utm_dtype)
utm_data['easting'] = 500000.0 + np.random.randn(n) * 1000
utm_data['northing'] = 4649776.0 + np.random.randn(n) * 1000
utm_data['zone'] = 33
utm_data['isNorthernHemisphere'] = 1

input_buffer = device.create_buffer(
    element_count=n,
    resource_type_layout=kernel.reflection.inputBuffer,
    usage=spy.BufferUsage.shader_resource,
    data=utm_data.view(np.uint8).reshape(-1),
)

output_buffer = device.create_buffer(
    element_count=n,
    resource_type_layout=kernel.reflection.outputBuffer,
    usage=spy.BufferUsage.unordered_access,
)


kernel.dispatch(
	thread_count=[n, 1, 1], 
	inputBuffer=input_buffer, 
	outputBuffer=output_buffer
)

result = output_buffer.to_numpy().view(coords_dtype)
print(result['x'][:5], result['y'][:5])
