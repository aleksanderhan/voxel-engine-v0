# voxel-engine-v0
GPU Voxel Renderer + Streaming World (wgpu)

![Demo](assets/demo.gif)

This project is a real-time voxel world renderer built around **wgpu** (WebGPU-native graphics API in Rust) and **WGSL** (WebGPU Shading Language). The core idea is:

- Stream a chunked voxel world around the camera.
- Ray-trace the world on the GPU using a sparse data structure.
- Layer atmosphere and lighting (fog, clouds, godrays).
- Present the final image with a lightweight fullscreen blit + tiny FPS HUD.

It’s structured like a minimal “engine loop” (winit) + a rendering backend (wgpu) + world/streaming systems.

---

## What you get

- **GPU ray tracing of voxels** (compute shader), with lighting and volumetrics.
- **Sparse voxel octree (SVO)** traversal with **ropes** for fast neighbor stepping.
- **Chunk streaming** driven by camera position and forward direction.
- **Macro-occupancy acceleration** (coarse empty-space skipping inside chunks).
- **Toroidal clipmap heightfield** for terrain/ground intersection & shading.
- **Multi-pass GPU pipeline**: primary → godray → composite → blit.
- **GPU timestamp profiling** when supported (TIMESTAMP_QUERY).
- **Minimal FPS overlay** rendered in the final blit shader.

---

## High-level architecture

### App / main loop (`src/app/mod.rs`)
The app runs a classic `winit` event loop:

- `AboutToWait` requests a redraw every iteration (continuous rendering).
- `RedrawRequested` performs **the entire frame** (update + render).
- Resize events reconfigure the swapchain and recreate internal textures.

Frame steps (in order):

1. **Integrate input → camera** (`camera.integrate_input`)
2. **Streaming update**: decide which chunks should be loaded/updated
3. **Clipmap update (CPU)**: compute per-level origins/offsets and height patches
4. **Write camera uniform** (view/projection matrices, params, grid info)
5. **Write overlay uniform** (packed digits + HUD layout)
6. **Apply chunk uploads** (batched buffer writes)
7. **Acquire swapchain frame**
8. **Encode GPU passes** (compute + blit)
9. **Submit + poll + present**
10. **Optional profiling readback**

---

## Rendering pipeline (GPU)

The renderer is compute-driven, writing into intermediate textures, then presenting with a simple render pass.

### Pass 1: Primary (compute)
- Writes:
  - `color_tex` (HDR color, **RGBA16F** / “half-float HDR”)
  - `depth_tex` (scene depth proxy, **R32F**)
- Uses:
  - Camera + scene buffers
  - Chunk grid + chunk metadata
  - SVO nodes + rope links
  - Macro occupancy + column info (extra acceleration data)
  - Clipmap params + clipmap height texture array

### Pass 2: Godrays (compute, ping-pong temporal)
- Uses a history texture (ping/pong) with temporal accumulation.
- Outputs a godray buffer (also HDR).
- Uses a sampler for filtered history reads.

### Pass 3: Composite (compute)
- Combines primary color + godrays into a final **full-resolution output** texture.
- Includes depth-aware upsampling and sharpening for godrays.
- Performs tonemapping (filmic curve), bloom-ish bright extraction, and grading.

### Pass 4: Blit (render)
- Fullscreen triangle.
- Samples the final output texture and draws into the swapchain.
- Renders a tiny FPS HUD (3×5 digit font) only inside a small screen rect.

---

## World representation + traversal

### Chunked world
The world is streamed in **chunks** around the camera. A GPU “chunk grid” maps grid cells to resident chunk slots.

Each chunk on GPU has:

- `ChunkMetaGpu`: origin, node arena base/count, macro base, column-info base
- Chunk grid entry: maps local grid index → chunk slot or INVALID

### Sparse voxel octree (SVO)
Voxels are stored in an SVO node arena (`NodeGpu`). Each node holds:

- `child_base` + `child_mask` for compact child addressing
- `material`
- `key` encoding the node’s level and coordinates (used to reconstruct bounds)

Traversal uses:

- **AABB** (Axis-Aligned Bounding Box) slab tests for entry/exit intervals
- **Ropes** (`NodeRopesGpu`) to jump to neighboring nodes when exiting a leaf cube
- **Sparse descent** that can return:
  - a real leaf node (explicit)
  - an implicit “air leaf” from a missing child (virtual cube), with an anchor to continue from

### Macro occupancy (coarse empty skipping)
Inside each chunk, a coarse **8×8×8** macro grid is stored as bits (512 bits = 16 `u32` words per chunk). This allows fast skipping of empty macro-cells using a 3D **DDA** (Digital Differential Analyzer) grid march before doing expensive leaf traversal.

### Column info acceleration (grass probing)
A compact 64×64 per-chunk “column-top map” packs `(y, material)` per (x,z) column into `u16` entries. It’s used to cheaply decide where grass blades might be, without doing full voxel traversal everywhere.

---

## Clipmap terrain (heightfield)

A toroidal clipmap stores height patches in a 2D array texture:

- Format: **R16Float** (half bandwidth vs R32F)
- Layout: `texture_2d_array`, one layer per clipmap level
- CPU updates decide which patches to upload each frame

The shader samples the clipmap using:

- Per-level origin in meters
- Per-level cell size
- Per-level toroidal offsets in texels

### Important ordering note (the “clipmap fix”)
Clipmap texture uploads and clipmap uniform updates must be encoded **before** the compute pass they affect. The code ensures the clipmap patch uploads + uniform write occur in the correct order so uniforms (origin/offset) cannot get ahead of texture content.

---

## Profiling

Two layers of profiling exist:

- **CPU-side frame profiling**: measures camera update, streaming, encoding, submit, present, etc.
- **GPU timestamps** (if supported): measures primary, godray, composite, and blit pass times via `TIMESTAMP_QUERY`.

When GPU timestamps are enabled, the renderer resolves timestamps into a buffer and maps it for readback, converting timestamp ticks to milliseconds.

---

## Project layout (relevant parts)

- `src/app/`  
  Main loop orchestration, resize handling, per-frame sequencing.
- `src/render/`  
  GPU types, resources, shader bundling, renderer state (pipelines/buffers/textures/bind groups).
- `src/shaders/`  
  WGSL shader modules (common utilities + raytracing modules + clipmap + blit).
- `src/streaming/`  
  ChunkManager and upload budgeting (CPU→GPU streaming).
- `src/world/`  
  WorldGen / procedural world source.

---

## Rendering data formats (quick cheat sheet)

- **HDR color**: `Rgba16Float`
- **Depth proxy**: `R32Float`
- **Clipmap height**: `R16Float` array texture
- **GPU scene buffers**: storage buffers (`NodeGpu`, `ChunkMetaGpu`, macro occupancy bits, rope links, column info)

---

## Notes & extension points

- The renderer is intentionally **compute-first**: most “rendering” logic lives in WGSL compute entry points.
- The scene bind group is shared across passes where possible; specialized bind groups are used for ping-pong textures (godrays/composite).
- Chunk uploads are aggressively batched (merged adjacent regions, contiguous meta runs) to reduce `queue.write_buffer` calls.

Common next steps:
- Add proper LOD selection for clipmap sampling (currently it can be fixed-level).
- Add a material system and/or voxel editing.
- Add denoising/temporal filtering for primary pass output.
- Expose profiler output in an on-screen overlay.

---
