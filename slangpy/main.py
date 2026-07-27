import slangpy as spy
import pathlib
import numpy as np

# Create a SlangPy device and use the local folder for Slang includes
device = spy.create_device(include_paths=[
        pathlib.Path(__file__).parent.absolute(),
])

# Load the module
module = spy.Module.load_from_file(device, "example.slang")

# Create two 2D tensors of size 16x16
image_1 = spy.Tensor.empty(device, dtype=module.Pixel, shape=(16, 16))
image_2 = spy.Tensor.empty(device, dtype=module.Pixel, shape=(16, 16))

# Populate the first tensor using a cursor
cursor_1 = image_1.cursor()
for x in range(16):
    for y in range(16):
        cursor_1[x + y * 16].write({
            'r': (x + y) / 32.0,
            'g': 0,
            'b': 0,
        })
cursor_1.apply()

# Populate the second tensor directly from a NumPy array
image_2.copy_from_numpy(0.1 * np.random.rand(16 * 16 * 3).astype(np.float32))

# Call the module's add function
result = module.add(image_1, image_2)

# Pre-allocate the result tensor
result = spy.Tensor.empty(device, dtype=module.Pixel, shape=(16, 16))
module.add(image_1, image_2, _result=result)

# Read and print pixel data using a cursor
result_cursor = result.cursor()
for x in range(16):
    for y in range(16):
        pixel = result_cursor[x + y * 16].read()
        print(f"Pixel ({x},{y}): {pixel}")

# Display the result with tev (https://github.com/Tom94/tev)
tex = device.create_texture(
    data=result.to_numpy(),
    width=16,
    height=16,
    format=spy.Format.rgb32_float
)
spy.tev.show(tex)