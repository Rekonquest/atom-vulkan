# atom-vk → Atom-Intel-Driver stage map

The atom-vk API is the user-space half of the driver's stacking order
(`Atom-Intel-Driver/docs/STACKING_ORDER.md`). Every operation names the
earliest driver stage that can implement it. The `atom-vk-arc` backend
fails closed with these stage names until they land — nothing silently
falls back.

## Backends (the constitution holds; the vendor is a generator swap)

| Backend | Vendor path | Crates | Status |
|---|---|---|---|
| `atom-vk-wgpu` | wgpu 26 → Vulkan (RTX 5070 Ti) | wgpu+pollster+bytemuck, gpu-crate only | verified, parity 1 LSB |
| `atom-vk-gl` | **OpenGL 4.3 compute, raw WGL/GL FFI — NVIDIA's own driver compiles the GLSL** | **ZERO external crates** | verified, parity 1 LSB (mean 0.002) |
| `atom-vk-arc` | Atom-Intel-Driver UAPI (stages 5–7) | zero-dep, fail-closed | walking skeleton |

Why GL and not raw Vulkan/CUDA: raw Vulkan consumes SPIR-V and CUDA
consumes PTX — both are compiler-walled (no zero-dep WGSL→SPIR-V/PTX
exists). OpenGL is the only vendor API where **the driver itself
compiles shader text** — the painted-wall answer for a zero-crate
vendor-native path. Same backend runs on AMD/Intel, including the A380
on stock drivers. Field note: NVIDIA's GL driver rejects the
GL_COPY_READ/WRITE_BUFFER target enums (INVALID_ENUM) on this machine;
SHADER_STORAGE + ARRAY_BUFFER are legal copy targets and used instead.

| atom-vk operation | Vulkan counterpart | Driver stage | Driver state |
|---|---|---|---|
| `create_buffer` | `vkCreateBuffer` | 6 — render/compute UAPI over Stage 4 LMEM allocator + GGTT | memory foundations landed (4V); no userspace boundary yet |
| `write_buffer` | mapped write | 6 — same UAPI; CPU writes into mapped LMEM page (proven 4P/4R) | blocked on UAPI |
| `read_buffer` | mapped read | 6 — same UAPI; BAR2 CPU reads (proven 4C–4E, 4M–4O) | blocked on UAPI |
| `create_storage_image` | `vkCreateImage` | 6 — tiled/linear image object over the same allocator | blocked on UAPI |
| `create_shader` | `vkCreateShaderModule` | 7 — user-space runtime: WGSL/SPIR-V → Xe-LP binary | not started |
| `create_pipeline` | `vkCreateComputePipelines` | 7 — pipeline state + binding table | not started |
| `submit_and_wait` | `vkQueueSubmit` + fence wait | 5 — execution foundations: command queues, doorbells, fences | **next driver layer after memory foundations** |

## What the game proved about this API

`demo-arena` runs atom-arena through atom-vk (wgpu backend, RTX 5070 Ti):

- **Parity vs the zero-dep CPU renderer: max channel diff 1 LSB, PASS,
  on BOTH GPU backends** — wgpu (mean 0.032) and NVIDIA-native GL (mean
  0.002). The API surface is information-lossless for this workload.
- Frame cost (fences + double-buffered readback, 600-frame legs):

| Resolution | wgpu full | **GL full** | GL submit-only | CPU (zero-dep) |
|---|---|---|---|---|
| 480×270 | 2,279 fps | **6,466 fps** | 25,204 fps | 143 fps |
| 960×540 | 622 fps | **1,113 fps** | 9,255 fps | 37 fps |
| 1920×1080 | 187.5 fps | **303 fps** | 2,564 fps | 9.5 fps |

  The zero-crate GL path is 1.6–2.8× faster than wgpu on full frames —
  the SSBO map readback on NVIDIA beats the staging+poll dance. The
  remaining cost is the readback itself; a swapchain present is the v2
  lever.
- The API now carries the fence primitive explicitly (`submit` →
  `Fence`, `wait_fence`) — the exact object the driver's Stage 5
  (execution foundations) builds. The client contract per frame:
  `write_buffer` ×3, `submit` with `Bind + Dispatch +
  CopyImageToBuffer`, deferred `read_buffer` on the previous target.
  That is the entire contract the Arc UAPI must honor to run this game
  on the A380.

## The smallest useful Arc milestone

When Stage 5 (execution foundations) and a minimal Stage 6 UAPI exist,
the first real end-to-end target is **one dispatch of this game's
shader**: uniform + 2 storage buffers in, rgba8 image out, one copy
back. Everything above Stage 5 (doorbell, fence) is already exercised
by that shape. The WGSL → Xe-LP question (Stage 7) is the only
long-lead item; a hand-written Xe-LP micro-kernel of `map(p)` would
unblock the milestone before a real compiler exists.
