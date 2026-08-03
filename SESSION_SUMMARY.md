# GIS Compute Shader Pipeline: Debugging Session Summary

## Goal

Build a GPU-accelerated geospatial coordinate transformation pipeline (MGRS/UTM
WGS84 → 7-parameter Helmert shift → EVRF2007 Mercator projection) using a Slang
compute shader, driven first by a native C++23/`slang-gfx` harness and later by
the Python SlangPy bindings, on an Intel UHD Graphics 620 (Whiskey Lake) iGPU
under WSL2/Vulkan.

## Part 1 — C++23 / slang-gfx Native Harness

The initial approach used Slang's `gfx` C++ wrapper API directly. This phase
was dominated by **API-surface guessing failures**: repeated attempts to
guess struct field names and function signatures (`IShaderProgram::Desc`
fields, `ICommandQueue::QueueType` values, fence/sync APIs, `bindPipeline`
semantics) against an SDK version whose actual headers were never consulted
first. Each guess produced a new, different compiler error.

**The fix that broke the cycle:** stop guessing and `grep` the actual
installed `slang-gfx.h` header directly. Once real signatures were pulled
from the header:

- `ICommandQueue::QueueType` only has one enumerator, `Graphics` — there is
  no `Compute` queue type in this SDK; compute dispatches go through the
  graphics queue.
- `IComputeCommandEncoder::bindPipeline()` **returns** the transient root
  shader object to bind resources on — no separate `createShaderObject` /
  reflection step is needed.
- Synchronization is `queue->executeCommandBuffer(...)` + `queue->waitOnHost()`
  — no fence object required for a simple blocking dispatch.
- `IShaderProgram::Desc`'s real field is `slangGlobalScope`, confirmed by
  reading the struct definition directly.

**Build/link issues, also resolved via direct inspection rather than guessing:**

| Symptom | Root cause | Fix |
|---|---|---|
| `undefined reference to gfxCreateDevice` | `gfxCreateDevice` lives in `libgfx.so`, a separate shared library from `libslang.so` — not merged in as assumed | Added `find_library(SLANG_GFX_LIBRARY NAMES gfx ...)` and linked it |
| `SLANG_INCLUDE_DIR-NOTFOUND` | `slang.h` lives at `include/slang/slang.h`, not flat under `include/` | Fixed `PATH_SUFFIXES` to `include/slang` |
| `Input buffer mapping failed` at runtime | `inputBuffer` was created with `MemoryType::DeviceLocal`, which isn't host-mappable; `map()`/`unmap()` require a host-visible heap | Changed to `MemoryType::Upload` for the buffer being written from the CPU (confirmed against the real `MemoryType` enum: `DeviceLocal`, `Upload`, `ReadBack`) |

Runtime linking also required `LD_LIBRARY_PATH` pointed at the SDK's `lib/`
directory, since `libgfx.so`/`libslang.so` use versioned SONAMEs not on a
standard system path. Occasional `munmap_chunk(): invalid pointer` crashes
under WSL2/AdaptiveCPP (in an earlier SYCL side-exploration) were traced to a
conflict between WSL2's virtual OpenCL passthrough and the runtime's memory
allocator, fixed with `export ACPP_VISIBILITY_MASK="omp"`.

## Part 2 — Migration to SlangPy (Python)

Switched to `slangpy` for a faster iteration loop. New category of bugs
appeared, again solved by reading real source (docstrings, `.pyi` stubs, and
critically, the SDK's own **test suite**, which turned out to be the most
reliable source of truth in the whole session) rather than guessing:

1. **`create_buffer()` `TypeError` with no specific culprit named.**
   Diagnosed by comparing against `test_buffer_from_resource_type_layout.py`
   and `test_buffer_cursor.py`:
   - `resource_type_layout=kernel.reflection.inputBuffer` (a raw
     `ReflectionCursor`) was actually **correct** — the official test uses
     the identical pattern.
   - The real problem was `data=utm_data`, a **structured/record numpy
     dtype**. SlangPy's `data=` kwarg only accepts flat, homogeneous-dtype
     arrays (confirmed via `test_apply_changes`'s
     `data=np.zeros(stride * count, dtype=np.uint8)` pattern). Fix:
     `data=utm_data.view(np.uint8).reshape(-1)`.

2. **`Expected a TypeLayoutReflection of a structured buffer for
   'resource_type_layout'`.** The shader wrapped its buffers in
   `ParameterBlock<StructuredBuffer<T>>`; SlangPy's `create_buffer()` wants
   the reflection of the `StructuredBuffer` itself. Fix: removed the
   `ParameterBlock` wrapper, using plain `public StructuredBuffer<T>` /
   `public RWStructuredBuffer<T>` globals instead — which also matches how
   the native C++ harness binds these same buffers.

3. **`use of uninitialized variable 'c'` compiler warnings** on every field
   access into a `TransformConstants` struct with default member
   initializers. In Slang (unlike C++), a bare local declaration
   (`TransformConstants c;`) does **not** implicitly invoke the constructor
   that applies default member initializers. Fix: explicit construction,
   `TransformConstants c = TransformConstants();`.

4. **Confirmed via a minimal isolated test** (single-`float` struct field,
   structurally mirroring the real shader) that the dispatch/buffer-binding
   mechanism for global-scope (non-parameter) `StructuredBuffer` declarations
   works correctly when bound via direct `kernel.dispatch(..., inputBuffer=...,
   outputBuffer=...)` kwargs — this eliminated an entire category of
   suspicion before moving on.

## Part 3 — The fp64 Hardware Limitation

With the binding/reflection/shader-init bugs all fixed, the real shader
still produced **all-zero output**, including a sentinel-only
`outputBuffer[id].x = 12345.0;` write with all real math removed. This was
diagnosed methodically rather than assumed:

1. **`device.has_feature(spy.Feature.double)` → `True`** — ruled out the
   simplest explanation (missing feature flag).
2. **Minimal isolated double-only test** (`double_minimal.slang` /
   `main_double_minimal.py`, mirroring the earlier confirmed-working `float`
   minimal test field-for-field) — **still zero**, with correct reflection
   (`stride = 8`, matching numpy `itemsize`). This ruled out struct
   complexity/layout entirely.
3. **Vulkan validation layers** (`enable_debug_layers=True` on the real
   `create_device()` API, after discovering `VK_INSTANCE_LAYERS`/
   `INTEL_DEBUG` env vars weren't reaching SlangPy's explicitly-constructed
   Vulkan instance) — **zero output, but also zero validation complaints.**
   The API usage is spec-valid; the failure is in numerical execution, a
   layer validation doesn't check.
4. **Bit-packed double test** (`bitpacked_double.slang` /
   `main_bitpacked_double.py`) — doubles split into `uint32` pairs on the
   host, reassembled via `asdouble()`/`asuint()` in-shader. This eliminated
   storage-buffer layout as a variable entirely (every stored field is
   `uint32`, identical to types that work correctly elsewhere) — **still
   zero.** This isolated the failure specifically to the `asdouble`/`asuint`
   double-precision reconstruction/arithmetic path itself.

**Conclusion:** the Intel UHD Graphics 620 (Gen9.5) has no native
double-precision execution hardware. `Feature.double` reporting `True`
reflects the existence of Mesa ANV's software/NIR fp64 emulation layer for
spec compliance, not a correctness guarantee — and empirically, across four
independent angles of attack, that emulation path does not produce correct
results on this hardware for this workload. This is a hardware/driver
limitation, not a bug in the shader, the Python code, or the buffer layout.

## Resolution: fp32 Camera-Relative Jacobian

Given the fp64 dead end, the working path forward avoids double precision on
the GPU entirely:

- **Host (Python/numpy, `float64`):** compute a local anchor point and a 2×2
  Jacobian matrix (via numerical central-difference differentiation of the
  full projection chain) once per camera/viewport position.
- **GPU (Slang, `float` only):** receive small `float` deltas relative to
  the anchor and apply the linear Jacobian approximation — no trig, no
  double, anywhere in the shader.

This mirrors the original fp32-vs-fp64 precision/range discussion from the
very start of this thread (global-scale coordinates in `float` lose
sub-meter precision; the Camera-Relative technique sidesteps this by working
in small local deltas) — except here the constraint driving the choice
turned out to be hardware capability rather than just numerical precision.

## Files Produced This Session

| File | Purpose |
|---|---|
| `main.cpp` | Native C++23 harness using `slang-gfx` directly |
| `CMakeLists.txt` | Build config linking `libslang.so` + `libgfx.so` |
| `spatial_transform.slang` | Real GIS shader (Helmert/Mercator), with `TransformConstants` and `ParameterBlock` fixes applied |
| `main_slangpy.py` | SlangPy Python harness for the real shader |
| `double_minimal.slang` / `main_double_minimal.py` | Minimal isolated fp64 hardware test — reusable on future hardware |
| `bitpacked_double.slang` / `main_bitpacked_double.py` | Bit-packed double workaround/diagnostic |
| `fp32_jacobian.slang` / `main_fp32_jacobian.py` | **Recommended path forward** — fp32-only GPU pipeline |

## Key Lesson

Every extended debugging cycle in this session was caused by the same
underlying mistake: guessing an API surface instead of reading the actual
installed source (headers, `.pyi` stubs, or — most productively — the SDK's
own test suite) before writing code against it. Once that discipline was
applied consistently, each remaining bug was found and fixed in one or two
turns rather than five or six.
