//! atom-vk-wgpu — the reference GPU backend. The CPU only submits
//! (command encoder + one queue submission); all execution is on the
//! GPU. This is the same shape the Arc backend will have once the
//! driver's Stage 5/6/7 land: build commands on the CPU, ring the
//! doorbell, wait on a fence.

use atom_vk::{BufferId, Command, Device, Fence, ImageId, PipelineId, PipelineLayout, ShaderId};
use atom_vk::{ShaderSource, VkError};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

struct Buffer {
    inner: wgpu::Buffer,
    size: u64,
}

struct Image {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    w: u32,
    h: u32,
    /// 256-aligned row-pitch staging area for readback; rows are then
    /// compacted GPU-side into the client's target buffer.
    staging: wgpu::Buffer,
    bpr: u32,
}

struct Pipeline {
    inner: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    layout: PipelineLayout,
}

struct State {
    buffers: Vec<Buffer>,
    images: Vec<Image>,
    shaders: Vec<wgpu::ShaderModule>,
    pipelines: Vec<Pipeline>,
    /// Readback staging pool, keyed by buffer id — a per-frame
    /// create/destroy cycle was the dominant cost of the naive path.
    readbacks: std::collections::HashMap<u32, wgpu::Buffer>,
    /// Fence registry: one flag per submission, set by the queue's
    /// work-done callback. `wait_fence` spins the device poll on it.
    fences: Vec<Arc<AtomicBool>>,
}

pub struct WgpuDevice {
    name: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    state: Mutex<State>,
}

impl WgpuDevice {
    pub fn new() -> Result<WgpuDevice, VkError> {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .map_err(|e| VkError::Backend(format!("no GPU adapter: {e}")))?;
            let info = adapter.get_info();
            let name = format!(
                "atom-vk wgpu on {} ({:?}, {:?})",
                info.name, info.device_type, info.backend
            );
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("atom-vk-wgpu"),
                    ..Default::default()
                })
                .await
                .map_err(|e| VkError::Backend(format!("request_device failed: {e}")))?;
            Ok(WgpuDevice {
                name,
                device,
                queue,
                state: Mutex::new(State {
                    buffers: Vec::new(),
                    images: Vec::new(),
                    shaders: Vec::new(),
                    pipelines: Vec::new(),
                    readbacks: std::collections::HashMap::new(),
                    fences: Vec::new(),
                }),
            })
        })
    }
}

impl Device for WgpuDevice {
    fn name(&self) -> &str {
        &self.name
    }

    fn create_buffer(&self, size: u64) -> Result<BufferId, VkError> {
        let inner = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vk-buffer"),
            size: size.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut s = self
            .state
            .lock()
            .map_err(|_| VkError::Backend("poisoned".into()))?;
        s.buffers.push(Buffer { inner, size });
        Ok((s.buffers.len() - 1) as BufferId)
    }

    fn write_buffer(&self, buf: BufferId, offset: u64, data: &[u8]) -> Result<(), VkError> {
        let s = self
            .state
            .lock()
            .map_err(|_| VkError::Backend("poisoned".into()))?;
        let b = s
            .buffers
            .get(buf as usize)
            .ok_or(VkError::Backend(format!("bad buffer id {buf}")))?;
        if offset + data.len() as u64 > b.size {
            return Err(VkError::Backend("write_buffer out of range".into()));
        }
        self.queue.write_buffer(&b.inner, offset, data);
        Ok(())
    }

    fn read_buffer(&self, buf: BufferId) -> Result<Vec<u8>, VkError> {
        let mut s = self
            .state
            .lock()
            .map_err(|_| VkError::Backend("poisoned".into()))?;
        let (src, size) = {
            let b = s
                .buffers
                .get(buf as usize)
                .ok_or(VkError::Backend(format!("bad buffer id {buf}")))?;
            (b.inner.clone(), b.size)
        };
        let staging = s.readbacks.entry(buf).or_insert_with(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vk-readback"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read"),
            });
        enc.copy_buffer_to_buffer(&src, 0, staging, 0, size);
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        // Drive the device only until THIS map resolves — never wait on
        // unrelated in-flight submissions (that would re-serialize the
        // pipelined frame loop). Spin-poll: the deferred-read data is
        // almost always already resident, so this resolves immediately.
        loop {
            self.device
                .poll(wgpu::PollType::Poll)
                .map_err(|e| VkError::Backend(format!("poll failed: {e}")))?;
            match rx.try_recv() {
                Ok(res) => {
                    res.map_err(|e| VkError::Backend(format!("map failed: {e}")))?;
                    break;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    std::thread::yield_now();
                    continue;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Err(VkError::Backend("map channel dropped".into()));
                }
            }
        }
        let out = slice.get_mapped_range().to_vec();
        staging.unmap();
        Ok(out)
    }

    fn create_storage_image(&self, w: u32, h: u32) -> Result<ImageId, VkError> {
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vk-image"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let bpr = (w * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vk-image-staging"),
            size: u64::from(bpr) * u64::from(h),
            // texture copy lands here (DST), row compaction copies out (SRC)
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut s = self
            .state
            .lock()
            .map_err(|_| VkError::Backend("poisoned".into()))?;
        s.images.push(Image {
            tex,
            view,
            w,
            h,
            staging,
            bpr,
        });
        Ok((s.images.len() - 1) as ImageId)
    }

    fn create_shader(&self, src: ShaderSource) -> Result<ShaderId, VkError> {
        let module = match src {
            ShaderSource::Wgsl(text) => {
                self.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("vk-shader"),
                        source: wgpu::ShaderSource::Wgsl(text.into()),
                    })
            }
            ShaderSource::Glsl(_) => {
                return Err(VkError::Unsupported(
                    "wgpu backend consumes WGSL; GLSL belongs to the OpenGL backend",
                ));
            }
        };
        let mut s = self
            .state
            .lock()
            .map_err(|_| VkError::Backend("poisoned".into()))?;
        s.shaders.push(module);
        Ok((s.shaders.len() - 1) as ShaderId)
    }

    fn create_pipeline(
        &self,
        shader: ShaderId,
        layout: PipelineLayout,
    ) -> Result<PipelineId, VkError> {
        let s = self
            .state
            .lock()
            .map_err(|_| VkError::Backend("poisoned".into()))?;
        let module = s
            .shaders
            .get(shader as usize)
            .ok_or(VkError::Backend(format!("bad shader id {shader}")))?;

        let mut entries = Vec::new();
        if layout.uniform {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        if layout.storage_image {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            });
        }
        for i in 0..layout.storage_buffers {
            // the trailing `storage_writable` buffers are read_write
            let read_only = i < layout.storage_buffers - layout.storage_writable;
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 2 + i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vk-bgl"),
                entries: &entries,
            });
        let pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("vk-pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
        let inner = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("vk-pipeline"),
                layout: Some(&pl),
                module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let mut s = s;
        s.pipelines.push(Pipeline { inner, bgl, layout });
        Ok((s.pipelines.len() - 1) as PipelineId)
    }

    fn submit(&self, cmds: &[Command]) -> Result<Fence, VkError> {
        let mut s = self
            .state
            .lock()
            .map_err(|_| VkError::Backend("poisoned".into()))?;
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vk-submit"),
            });
        let mut bind_group: Option<wgpu::BindGroup> = None;
        let mut bound_pipeline: Option<usize> = None;
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vk-pass"),
            timestamp_writes: None,
        });
        for cmd in cmds {
            match cmd {
                Command::Bind {
                    pipeline,
                    uniform,
                    storages,
                    image,
                } => {
                    let p = s
                        .pipelines
                        .get(*pipeline as usize)
                        .ok_or(VkError::Backend(format!("bad pipeline id {pipeline}")))?;
                    let mut entries = Vec::new();
                    if p.layout.uniform {
                        let u = uniform
                            .and_then(|id| s.buffers.get(id as usize))
                            .ok_or(VkError::Backend("bind: missing/bad uniform".into()))?;
                        entries.push(wgpu::BindGroupEntry {
                            binding: 0,
                            resource: u.inner.as_entire_binding(),
                        });
                    }
                    if p.layout.storage_image {
                        let img = image
                            .and_then(|id| s.images.get(id as usize))
                            .ok_or(VkError::Backend("bind: missing/bad image".into()))?;
                        entries.push(wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&img.view),
                        });
                    }
                    for (i, id) in storages.iter().enumerate() {
                        let b = s
                            .buffers
                            .get(*id as usize)
                            .ok_or(VkError::Backend(format!("bind: bad storage id {id}")))?;
                        entries.push(wgpu::BindGroupEntry {
                            binding: 2 + i as u32,
                            resource: b.inner.as_entire_binding(),
                        });
                    }
                    bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("vk-bg"),
                        layout: &p.bgl,
                        entries: &entries,
                    }));
                    bound_pipeline = Some(*pipeline as usize);
                }
                Command::Dispatch { groups_x, groups_y } => {
                    let pi =
                        bound_pipeline.ok_or(VkError::Backend("dispatch before bind".into()))?;
                    let bg = bind_group
                        .as_ref()
                        .ok_or(VkError::Backend("dispatch before bind".into()))?;
                    pass.set_pipeline(&s.pipelines[pi].inner);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(*groups_x, *groups_y, 1);
                }
                Command::Barrier => {
                    // WebGPU guarantees storage-write visibility between
                    // dispatch calls in a submission — the barrier is
                    // implicit here. (GL needs it spelled out; see vk-gl.)
                }
                Command::CopyImageToBuffer { image, buffer } => {
                    let img = s
                        .images
                        .get(*image as usize)
                        .ok_or(VkError::Backend(format!("bad image id {image}")))?;
                    let dst = s
                        .buffers
                        .get(*buffer as usize)
                        .ok_or(VkError::Backend(format!("bad buffer id {buffer}")))?;
                    if u64::from(img.w) * u64::from(img.h) * 4 > dst.size {
                        return Err(VkError::Backend("copy target too small".into()));
                    }
                    // Row-pitch rule: texture copies need 256-aligned
                    // rows, client buffers are tight. Stage padded, then
                    // compact GPU-side row by row.
                    drop(pass);
                    enc.copy_texture_to_buffer(
                        wgpu::TexelCopyTextureInfo {
                            texture: &img.tex,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyBufferInfo {
                            buffer: &img.staging,
                            layout: wgpu::TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(img.bpr),
                                rows_per_image: Some(img.h),
                            },
                        },
                        wgpu::Extent3d {
                            width: img.w,
                            height: img.h,
                            depth_or_array_layers: 1,
                        },
                    );
                    for y in 0..img.h {
                        enc.copy_buffer_to_buffer(
                            &img.staging,
                            u64::from(y) * u64::from(img.bpr),
                            &dst.inner,
                            u64::from(y) * u64::from(img.w) * 4,
                            u64::from(img.w) * 4,
                        );
                    }
                    pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("vk-pass"),
                        timestamp_writes: None,
                    });
                }
            }
        }
        drop(pass);
        self.queue.submit([enc.finish()]);
        let flag = Arc::new(AtomicBool::new(false));
        {
            let flag = Arc::clone(&flag);
            self.queue.on_submitted_work_done(move || {
                flag.store(true, Ordering::Release);
            });
        }
        s.fences.push(flag);
        Ok(Fence((s.fences.len() - 1) as u64))
    }

    fn wait_fence(&self, fence: Fence) -> Result<(), VkError> {
        let flag = {
            let s = self
                .state
                .lock()
                .map_err(|_| VkError::Backend("poisoned".into()))?;
            s.fences
                .get(fence.0 as usize)
                .cloned()
                .ok_or(VkError::Backend(format!("bad fence id {}", fence.0)))?
        };
        while !flag.load(Ordering::Acquire) {
            self.device
                .poll(wgpu::PollType::Poll)
                .map_err(|e| VkError::Backend(format!("poll failed: {e}")))?;
            std::thread::yield_now();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // GPU-touching tests live in the demo-arena parity run; this crate
    // stays compile-clean on machines without a GPU.
    #[test]
    fn compiles() {}
}
