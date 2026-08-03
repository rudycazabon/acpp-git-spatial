import slangpy as spy
import numpy as np
import pathlib

device = spy.create_device(include_paths=[pathlib.Path(__file__).parent.absolute()])
program = device.load_program(module_name="fp32_jacobian.slang", entry_point_names=["computeMain"])
kernel = device.create_compute_kernel(program=program)

# ---------------------------------------------------------------------------
# Host-side fp64 math: this is the ONLY place double precision is used.
# Replace transform_chain() with your real UTM -> Helmert -> Mercator pipeline;
# it's a placeholder linear stand-in here so the file is runnable standalone.
# ---------------------------------------------------------------------------
def transform_chain(easting: float, northing: float) -> tuple[float, float]:
    # Placeholder -- swap in the real WGS84->EVRF2007 Mercator pipeline.
    scale_x, scale_y = 1.0002, 1.0005
    return easting * scale_x, northing * scale_y

def compute_anchor_and_jacobian(anchor_e: float, anchor_n: float, delta: float = 1.0):
    base_x, base_y = transform_chain(anchor_e, anchor_n)

    ex_plus_x, ex_plus_y   = transform_chain(anchor_e + delta, anchor_n)
    ex_minus_x, ex_minus_y = transform_chain(anchor_e - delta, anchor_n)
    dX_dE = (ex_plus_x - ex_minus_x) / (2.0 * delta)
    dY_dE = (ex_plus_y - ex_minus_y) / (2.0 * delta)

    nx_plus_x, nx_plus_y   = transform_chain(anchor_e, anchor_n + delta)
    nx_minus_x, nx_minus_y = transform_chain(anchor_e, anchor_n - delta)
    dX_dN = (nx_plus_x - nx_minus_x) / (2.0 * delta)
    dY_dN = (nx_plus_y - nx_minus_y) / (2.0 * delta)

    return {
        "anchorMercatorX": np.float32(base_x),
        "anchorMercatorY": np.float32(base_y),
        "j_dX_dE": np.float32(dX_dE),
        "j_dX_dN": np.float32(dX_dN),
        "j_dY_dE": np.float32(dY_dE),
        "j_dY_dN": np.float32(dY_dN),
    }

# ---------------------------------------------------------------------------
# GPU-side data: pure float32 throughout.
# ---------------------------------------------------------------------------
local_dtype = np.dtype({
    'names':   ['deltaEasting', 'deltaNorthing'],
    'formats': [np.float32, np.float32],
}, align=True)

n = 1024
anchor_easting, anchor_northing = 500000.0, 4649776.0

local_data = np.zeros(n, dtype=local_dtype)
local_data['deltaEasting']  = (np.random.randn(n) * 1000).astype(np.float32)
local_data['deltaNorthing'] = (np.random.randn(n) * 1000).astype(np.float32)

jacobian = compute_anchor_and_jacobian(anchor_easting, anchor_northing)

print("numpy itemsize:", local_dtype.itemsize)
print(kernel.reflection.inputBuffer)  # confirm stride matches before creating the buffer

input_buffer = device.create_buffer(
    element_count=n,
    resource_type_layout=kernel.reflection.inputBuffer,
    usage=spy.BufferUsage.shader_resource,
    data=local_data.view(np.uint8).reshape(-1),
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
    **jacobian,
)

coords_dtype = np.dtype({'names': ['x', 'y'], 'formats': [np.float32, np.float32]}, align=True)
result = output_buffer.to_numpy().view(coords_dtype)
print(result['x'][:5], result['y'][:5])
