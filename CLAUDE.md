# CLAUDE.md

Project context for Claude Code. Read this before touching build files,
shader code, or Python harnesses in this repo.

## What this project is

A GPU compute pipeline for geospatial coordinate transformation:
MGRS/UTM (WGS84) → 7-parameter Helmert datum shift → EVRF2007 Mercator
projection, implemented as a Slang compute shader with two host drivers:
a native C++23/`slang-gfx` harness and a Python/SlangPy harness.

## Environment

- WSL2, Ubuntu 24.04
- Vulkan SDK at `/home/rudycazabon/Libraries/vulkan-sdk/1.4.350.1/x86_64`
- GPU: Intel UHD Graphics 620 (Whiskey Lake, Gen9.5) — **see "Known hardware
  limitation" below before writing any shader touching `double`.**
- Python harness uses `uv run <script>.py` from `~/acpp-gis-spatial/slangpy`
  or `~/acpp-gis-spatial/debug`.
- `libgfx.so` / `libslang.so` aren't on a standard library path:
  `export LD_LIBRARY_PATH=/home/rudycazabon/Libraries/vulkan-sdk/1.4.350.1/x86_64/lib:$LD_LIBRARY_PATH`
- If AdaptiveCPP/SYCL work resumes and hits `munmap_chunk(): invalid pointer`,
  it's a WSL2/OpenCL-passthrough driver conflict, not a code bug:
  `export ACPP_VISIBILITY_MASK="omp"`

## Hard rule: verify API signatures before writing code against them

This codebase burned enormous time on guessed `slang-gfx` C++ struct fields
and SlangPy kwargs that turned out wrong. **Never guess a signature.**
Before calling any unfamiliar `gfx::`/`slangpy` API:

1. `grep` the real header/stub first:
   - C++: `/home/rudycazabon/Libraries/vulkan-sdk/1.4.350.1/x86_64/include/slang/slang-gfx.h`
   - Python: `<venv>/lib/python3.12/site-packages/slangpy/__init__.pyi`
2. If the stub/header doesn't resolve it, check the SlangPy test suite —
   it was the single most reliable source of truth in this project:
   `<venv>/lib/python3.12/site-packages/slangpy/tests/`
3. Only fall back to a docstring or educated guess if both of the above come
   up empty, and flag it explicitly as unverified in a code comment.

## Verified `slang-gfx` (C++) gotchas — don't relitigate these

- `gfxCreateDevice` lives in `libgfx.so`, a **separate** shared library from
  `libslang.so`. Both must be linked.
- `slang.h` is at `include/slang/slang.h`, not flat under `include/`.
- `ICommandQueue::QueueType` has only `Graphics` — there is no `Compute`
  queue type in this SDK.
- `IComputeCommandEncoder::bindPipeline()` **returns** the root shader
  object to bind resources on — don't separately call `createShaderObject`.
- Sync is `queue->executeCommandBuffer(...)` + `queue->waitOnHost()` — no
  fence needed for a simple blocking dispatch.
- `IShaderProgram::Desc`'s field is `slangGlobalScope`.
- Buffers you `map()`/`unmap()` from the host **must** use
  `MemoryType::Upload`, not `DeviceLocal` (which isn't host-mappable).

## Verified SlangPy (Python) gotchas — don't relitigate these

- `create_buffer(data=...)` only accepts **flat, homogeneous-dtype**
  numpy arrays. A structured/record dtype (named fields) silently fails
  overload resolution with a generic `TypeError`. Fix: reinterpret first —
  `structured_array.view(np.uint8).reshape(-1)`.
- `resource_type_layout=kernel.reflection.<fieldname>` (a raw
  `ReflectionCursor`) is the correct value to pass — don't navigate into
  `.type` or `.type_layout` first.
- Shader-side buffer fields must be plain `public StructuredBuffer<T>` /
  `public RWStructuredBuffer<T>` globals — **not** wrapped in
  `ParameterBlock<...>` — or `create_buffer()` rejects the reflection with
  `Expected a TypeLayoutReflection of a structured buffer`.
- Global-scope (non-entry-point-parameter) buffers/uniforms bind correctly
  via direct kwargs on `kernel.dispatch(..., inputBuffer=..., outputBuffer=...)`
  — confirmed empirically, don't second-guess this pattern.
- To enable real Vulkan/RHI validation diagnostics, use
  `spy.create_device(enable_debug_layers=True, ...)` — shell env vars like
  `VK_INSTANCE_LAYERS` and `INTEL_DEBUG` do **not** reach SlangPy's
  explicitly-constructed Vulkan instance.

## Slang language gotcha

A bare local struct declaration (`TransformConstants c;`) does **not**
implicitly invoke the constructor that applies default member initializers
in Slang, unlike C++. Always write `TransformConstants c = TransformConstants();`
or bind constants via an actual uniform/constant buffer instead of relying
on in-struct defaults.

## Known hardware limitation: no working native fp64

**Do not spend time debugging `double`-typed storage buffer reads/writes on
this hardware — it has been exhaustively confirmed broken, not a code bug.**

Evidence trail (see `SESSION_SUMMARY.md` for full detail):
- `device.has_feature(spy.Feature.double)` reports `True`, but is
  misleading — it reflects Mesa ANV's software/NIR fp64 emulation
  existing, not that it's correct.
- A minimal single-`double`-field buffer test (`double_minimal.slang` /
  `main_double_minimal.py`) — mirroring the confirmed-working `float`
  version field-for-field — silently returns all zeros.
- Vulkan validation (`enable_debug_layers=True`) reports **no errors**
  against this same test — the failure is in numerical execution, not API
  usage.
- A bit-packed workaround (`double` split into two `uint32`s host-side,
  reassembled with `asdouble()`/`asuint()` in-shader) **also** fails,
  ruling out storage-buffer layout and isolating the bug to the emulated
  arithmetic path itself.

Root cause: Intel UHD 620 (Gen9.5) has no native double-precision execution
hardware. This is a driver/hardware ceiling, not something fixable in this
codebase.

**If this project ever runs on different hardware** (discrete GPU,
datacenter-class fp64 hardware), re-run `main_double_minimal.py` first — if
`Got` matches `Expected`, native fp64 is safe to use there and the fp32
workaround below becomes unnecessary.

## Current direction: fp32 Camera-Relative Jacobian

Given the fp64 ceiling, the project's GPU-side code should never touch
`double`. The established pattern:

- **Host (Python/numpy, `float64` — fine, this runs on CPU):** compute a
  local anchor point and a 2×2 Jacobian matrix via numerical central-
  difference differentiation of the full WGS84→Helmert→Mercator chain.
- **GPU (Slang, `float` only):** receive small `float` deltas relative to
  the anchor, apply the linear Jacobian approximation. No trig, no double,
  anywhere in the shader.

Reference implementation: `fp32_jacobian.slang` / `main_fp32_jacobian.py`.
The `transform_chain()` placeholder in the Python host needs to be replaced
with the real projection math before this is production-ready — that's the
next concrete task.

## File index

| File | Purpose |
|---|---|
| `main.cpp` + `CMakeLists.txt` | Native C++23 harness (`slang-gfx`) |
| `spatial_transform.slang` | Real GIS shader — Helmert/Mercator, currently fp64-based |
| `main_slangpy.py` | SlangPy Python harness for the real shader |
| `double_minimal.slang` / `main_double_minimal.py` | Portable fp64-hardware-support test — rerun on new hardware |
| `bitpacked_double.slang` / `main_bitpacked_double.py` | Bit-packing diagnostic/workaround |
| `fp32_jacobian.slang` / `main_fp32_jacobian.py` | **Current recommended path** — fp32-only GPU pipeline, needs real math wired in |
| `SESSION_SUMMARY.md` | Full narrative of how each bug above was found and fixed |

## Immediate next task

Port the real Helmert/Mercator math (currently in `spatial_transform.slang`,
fp64) into `fp32_jacobian.slang`'s linear-approximation shape, and replace
`main_fp32_jacobian.py`'s placeholder `transform_chain()` with the actual
projection pipeline. Test against `double_minimal.slang`'s pattern of
`Expected` vs. `Got` comparison to validate numerical correctness within
the ~100km anchor-radius accuracy envelope discussed earlier in this
project's history.
