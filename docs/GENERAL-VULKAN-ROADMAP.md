# The General Vulkan Replacement — Roadmap and Honest Status

The goal: a general replacement for Vulkan, composed from atoms. This
document decomposes what "general" actually means and tracks what is
proven vs pending. Nothing here is claimed without a measured run.

## The decomposition

Vulkan's surface, stripped to operations (names are illusions):

1. **Compute dispatch** — buffers, one kernel, fences, readback.
2. **Graphics** — vertices → raster → framebuffer.
3. **Texture objects / samplers** — typed pixel arrays + filtering.
4. **Presentation** — a swapchain (frames to a window without CPU copies).
5. **Memory model** — allocation, binding, residency.
6. **Sync objects** — semaphores, events, pipeline barriers, timelines.
7. **Shader binary path** — SPIR-V ecosystem / vendor ISAs.

## Status (measured, this machine)

| Stage | Status | Evidence |
|---|---|---|
| 1. Compute | **DONE, both GPU backends** | parity 1 LSB; GL 6,466 fps / wgpu 2,279 fps full-frame at 480×270 |
| 2. Graphics | **DONE at demo scale, both backends** | cube rasterized on GPU, parity 100% within 2 LSB vs zero-dep CPU reference; GL **7,122 / 1,835 / 574 fps**, wgpu 2,464 / 852 / 244 fps at 480×270 / 960×540 / 1080p; CPU loses 4.1–7.0× |
| 3. Textures/samplers | **DONE** | **painted wall: a texture is an SSBO, a sampler is `combine()`** — bilinear filtering kernel-side, wrap-repeat, zero new API commands. Textured cube: parity 100% within 2 LSB, GL 6,114 / 1,451 / 464 fps, wgpu 2,427 / 783 / 229 fps; CPU loses 3.0–10.4× |
| 6a. Fences + pipeline barriers | **DONE** | `submit` → `Fence`, `wait_fence`, `Command::Barrier` in one submission |
| 4. Presentation | partial | readback + std-only GDI window (works, costs PCIe); true swapchain = v2 |
| 5. Memory model | pending | buffers exist; allocator/residency does not |
| 6b. Semaphores/timelines/multi-queue | pending | fences + barriers only |
| 7. Shader binary | pending | WGSL (wgpu) / GLSL (vendor driver) today; SPIR-V is a compiler wall, Xe-LP is Arc Stage 7 |

## What the graphics stage taught (currency, not vibes)

- **One-thread-per-triangle rasterization is WRONG-shaped for a GPU.**
  First generator: transform → bbox → atomicMin z → resolve, 3 kernels.
  Pixel-exact but 38 fps at 1080p — *slower than the CPU reference
  (86 fps)*. Twelve serial threads × ~2M px loops. Composted.
- **One-thread-per-pixel is the right generator at demo scale.**
  Every pixel loops all triangles; no z-buffer, no atomics, no clear
  pass. Same parity, 574 fps at 1080p, 7× CPU. (GLSL note: `packed` is
  a reserved word; NVIDIA rejects GL_COPY_READ/WRITE_BUFFER enums.)
- **The attention step (tile binning) is the generator for big meshes:**
  assign triangles to screen tiles first, so per-pixel triangle loops
  shrink from N to ~k. Same operation as cognitive attention — top-k
  resource arbitration — implemented when mesh size demands it.

## The order ahead

1. ~~Textures/samplers~~ **DONE** (SSBO + kernel-side `combine()` sampler).
2. True swapchain present (kills the readback cost — the last big
   structural win; GL: wglSwapBuffers blit path; wgpu: Surface on raw
   HWND).
3. Memory allocator + residency semantics.
4. Semaphores/timeline sync + multi-queue.
5. Attention/binning rasterizer for large meshes (the scalable
   graphics generator).
6. Shader-binary path (SPIR-V ingestion or the Xe-LP compiler for Arc).

Each stage lands verified or it doesn't land. `demo-raster` is the
template for stage N+1 harnesses: CPU reference → parity → bench →
commitment tag.
