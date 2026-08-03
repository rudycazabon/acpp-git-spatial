import slangpy as spy
import numpy as np
import pathlib

device = spy.create_device(include_paths=[pathlib.Path(__file__).parent.absolute()])
program = device.load_program(module_name="bitpacked_double.slang", entry_point_names=["computeMain"])
kernel = device.create_compute_kernel(program=program)

def double_to_uint32_pair(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reinterpret a float64 array's raw bits as uint64, then split into
    low/high uint32 halves matching HLSL/Slang's asdouble(lowbits, highbits)
    ordering (low bits first)."""
    as_uint64 = arr.astype(np.float64).view(np.uint64)
    low  = (as_uint64 & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    high = (as_uint64 >> np.uint64(32)).astype(np.uint32)
    return low, high

def uint32_pair_to_double(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    as_uint64 = low.astype(np.uint64) | (high.astype(np.uint64) << np.uint64(32))
    return as_uint64.view(np.float64)

# ---------------------------------------------------------------------------
# Flat scalar layout -- deliberately avoids uint2/vector types on the numpy
# side to sidestep any vector-alignment guessing; matches the flat uint
# fields in bitpacked_double.slang exactly.
# ---------------------------------------------------------------------------
packed_dtype = np.dtype({
    'names':   ['easting_lo', 'easting_hi', 'northing_lo', 'northing_hi', 'zone', 'isNorthernHemisphere'],
    'formats': [np.uint32, np.uint32, np.uint32, np.uint32, np.int32, np.int32],
}, align=True)

n = 1024
easting  = 500000.0 + np.random.randn(n) * 1000
northing = 4649776.0 + np.random.randn(n) * 1000

e_lo, e_hi = double_to_uint32_pair(easting)
n_lo, n_hi = double_to_uint32_pair(northing)

packed_data = np.zeros(n, dtype=packed_dtype)
packed_data['easting_lo']  = e_lo
packed_data['easting_hi']  = e_hi
packed_data['northing_lo'] = n_lo
packed_data['northing_hi'] = n_hi
packed_data['zone'] = 33
packed_data['isNorthernHemisphere'] = 1

print("numpy itemsize:", packed_dtype.itemsize)
print(kernel.reflection.inputBuffer)  # confirm stride matches before creating the buffer

input_buffer = device.create_buffer(
    element_count=n,
    resource_type_layout=kernel.reflection.inputBuffer,
    usage=spy.BufferUsage.shader_resource,
    data=packed_data.view(np.uint8).reshape(-1),
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

out_dtype = np.dtype({
    'names':   ['x_lo', 'x_hi', 'y_lo', 'y_hi'],
    'formats': [np.uint32, np.uint32, np.uint32, np.uint32],
}, align=True)

raw = output_buffer.to_numpy().view(out_dtype)
result_x = uint32_pair_to_double(raw['x_lo'], raw['x_hi'])
result_y = uint32_pair_to_double(raw['y_lo'], raw['y_hi'])

print(result_x[:5], result_y[:5])
