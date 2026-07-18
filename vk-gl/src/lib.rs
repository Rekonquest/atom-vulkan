//! atom-vk-gl — the vendor-native OpenGL compute backend. Zero external
//! crates: hand-rolled WGL/GL FFI (see `sys`). The vendor's own driver
//! (NVIDIA today, AMD/Intel tomorrow — including the A380 on stock
//! drivers) compiles the GLSL text. The CPU submits and fences; the GPU
//! executes — the same contract shape as the wgpu backend.
//!
//! Binding convention (shared with the GLSL codegen):
//!   0 = uniform block, 1.. = storage buffers in Bind order,
//!   7 = pixel SSBO (the storage image — fixed high slot so multi-buffer
//!   pipelines never collide). Images are plain shader-storage buffers
//!   of packed 0x00RRGGBB words — no texture objects, no row pitch, no
//!   swizzle, copies are one call.

#[cfg(windows)]
mod sys;

#[cfg(windows)]
pub use imp::GlDevice;

#[cfg(windows)]
mod imp {
    use crate::sys;
    use atom_vk::{
        BufferId, Command, Device, Fence, ImageId, PipelineId, PipelineLayout, ShaderId,
    };
    use atom_vk::{ShaderSource, VkError};
    use std::sync::Mutex;

    struct Buffer {
        id: sys::GLuint,
        size: u64,
    }

    struct Image {
        buf: sys::GLuint,
        w: u32,
        h: u32,
    }

    struct State {
        buffers: Vec<Buffer>,
        images: Vec<Image>,
        shaders: Vec<sys::GLuint>,
        programs: Vec<sys::GLuint>,
        fences: Vec<sys::GLsync>,
    }

    /// Single-thread affinity: the WGL context is current on the thread
    /// that created the device. Use it from that thread.
    pub struct GlDevice {
        name: String,
        ctx: sys::Context,
        state: Mutex<State>,
    }

    impl GlDevice {
        pub fn new() -> Result<GlDevice, VkError> {
            let ctx = sys::Context::new().map_err(VkError::Backend)?;
            let name = format!("atom-vk gl on {}", ctx.renderer);
            Ok(GlDevice {
                name,
                ctx,
                state: Mutex::new(State {
                    buffers: Vec::new(),
                    images: Vec::new(),
                    shaders: Vec::new(),
                    programs: Vec::new(),
                    fences: Vec::new(),
                }),
            })
        }

        /// The raw GL error state (diagnostics; 0 = GL_NO_ERROR).
        pub fn last_error(&self) -> u32 {
            unsafe { (self.ctx.fns.get_error)() }
        }

        fn lock(&self) -> Result<std::sync::MutexGuard<'_, State>, VkError> {
            self.state
                .lock()
                .map_err(|_| VkError::Backend("poisoned".into()))
        }

        fn buf_id(&self, s: &State, buf: BufferId) -> Result<sys::GLuint, VkError> {
            s.buffers
                .get(buf as usize)
                .map(|b| b.id)
                .ok_or(VkError::Backend(format!("bad buffer id {buf}")))
        }
    }

    impl Device for GlDevice {
        fn name(&self) -> &str {
            &self.name
        }

        fn create_buffer(&self, size: u64) -> Result<BufferId, VkError> {
            let f = &self.ctx.fns;
            let mut id = 0;
            unsafe {
                (f.gen_buffers)(1, &mut id);
                (f.bind_buffer)(sys::GL_SHADER_STORAGE_BUFFER, id);
                (f.buffer_data)(
                    sys::GL_SHADER_STORAGE_BUFFER,
                    size.max(16) as sys::GLsizeiptr,
                    std::ptr::null(),
                    sys::GL_DYNAMIC_COPY,
                );
            }
            let mut s = self.lock()?;
            s.buffers.push(Buffer {
                id,
                size: size.max(16),
            });
            Ok((s.buffers.len() - 1) as BufferId)
        }

        fn write_buffer(&self, buf: BufferId, offset: u64, data: &[u8]) -> Result<(), VkError> {
            let s = self.lock()?;
            let b = s
                .buffers
                .get(buf as usize)
                .ok_or(VkError::Backend(format!("bad buffer id {buf}")))?;
            if offset + data.len() as u64 > b.size {
                return Err(VkError::Backend("write_buffer out of range".into()));
            }
            let f = &self.ctx.fns;
            unsafe {
                (f.bind_buffer)(sys::GL_SHADER_STORAGE_BUFFER, b.id);
                // Orphan-then-fill: the streaming idiom — the driver
                // renames storage instead of stalling on a buffer the
                // GPU may still be reading from last frame's dispatch.
                (f.buffer_data)(
                    sys::GL_SHADER_STORAGE_BUFFER,
                    b.size as sys::GLsizeiptr,
                    std::ptr::null(),
                    sys::GL_STREAM_DRAW,
                );
                (f.buffer_sub_data)(
                    sys::GL_SHADER_STORAGE_BUFFER,
                    offset as sys::GLintptr,
                    data.len() as sys::GLsizeiptr,
                    data.as_ptr() as *const _,
                );
            }
            Ok(())
        }

        fn read_buffer(&self, buf: BufferId) -> Result<Vec<u8>, VkError> {
            let s = self.lock()?;
            let b = s
                .buffers
                .get(buf as usize)
                .ok_or(VkError::Backend(format!("bad buffer id {buf}")))?;
            let f = &self.ctx.fns;
            unsafe {
                (f.bind_buffer)(sys::GL_SHADER_STORAGE_BUFFER, b.id);
                let p = (f.map_buffer_range)(
                    sys::GL_SHADER_STORAGE_BUFFER,
                    0,
                    b.size as sys::GLsizeiptr,
                    sys::GL_MAP_READ_BIT,
                );
                if p.is_null() {
                    return Err(VkError::Backend("map_buffer_range failed".into()));
                }
                let out = std::slice::from_raw_parts(p as *const u8, b.size as usize).to_vec();
                (f.unmap_buffer)(sys::GL_SHADER_STORAGE_BUFFER);
                Ok(out)
            }
        }

        fn create_storage_image(&self, w: u32, h: u32) -> Result<ImageId, VkError> {
            let f = &self.ctx.fns;
            let mut id = 0;
            unsafe {
                (f.gen_buffers)(1, &mut id);
                (f.bind_buffer)(sys::GL_SHADER_STORAGE_BUFFER, id);
                (f.buffer_data)(
                    sys::GL_SHADER_STORAGE_BUFFER,
                    (u64::from(w) * u64::from(h) * 4) as sys::GLsizeiptr,
                    std::ptr::null(),
                    sys::GL_DYNAMIC_COPY,
                );
            }
            let mut s = self.lock()?;
            s.images.push(Image { buf: id, w, h });
            Ok((s.images.len() - 1) as ImageId)
        }

        fn create_shader(&self, src: ShaderSource) -> Result<ShaderId, VkError> {
            let text = match src {
                ShaderSource::Glsl(t) => t,
                ShaderSource::Wgsl(_) => {
                    return Err(VkError::Unsupported(
                        "gl backend consumes GLSL; WGSL belongs to the wgpu backend",
                    ));
                }
            };
            let f = &self.ctx.fns;
            unsafe {
                let id = (f.create_shader)(sys::GL_COMPUTE_SHADER);
                let ptr = text.as_ptr() as *const sys::GLchar;
                let len = text.len() as sys::GLint;
                (f.shader_source)(id, 1, &ptr, &len);
                (f.compile_shader)(id);
                let mut status = 0;
                (f.get_shaderiv)(id, sys::GL_COMPILE_STATUS, &mut status);
                if status == 0 {
                    let mut log_len = 0;
                    (f.get_shaderiv)(id, sys::GL_INFO_LOG_LENGTH, &mut log_len);
                    let mut log = vec![0i8; log_len.max(1) as usize];
                    let mut written = 0;
                    (f.shader_info_log)(id, log_len, &mut written, log.as_mut_ptr());
                    let msg = log
                        .iter()
                        .take(written as usize)
                        .map(|c| *c as u8 as char)
                        .collect::<String>();
                    return Err(VkError::Backend(format!("GLSL compile failed: {msg}")));
                }
                let mut s = self.lock()?;
                s.shaders.push(id);
                Ok((s.shaders.len() - 1) as ShaderId)
            }
        }

        fn create_pipeline(
            &self,
            shader: ShaderId,
            _layout: PipelineLayout,
        ) -> Result<PipelineId, VkError> {
            let s0 = self.lock()?;
            let shader_id = *s0
                .shaders
                .get(shader as usize)
                .ok_or(VkError::Backend(format!("bad shader id {shader}")))?;
            drop(s0);
            let f = &self.ctx.fns;
            unsafe {
                let prog = (f.create_program)();
                (f.attach_shader)(prog, shader_id);
                (f.link_program)(prog);
                let mut status = 0;
                (f.get_programiv)(prog, sys::GL_LINK_STATUS, &mut status);
                if status == 0 {
                    let mut log_len = 0;
                    (f.get_programiv)(prog, sys::GL_INFO_LOG_LENGTH, &mut log_len);
                    let mut log = vec![0i8; log_len.max(1) as usize];
                    let mut written = 0;
                    (f.program_info_log)(prog, log_len, &mut written, log.as_mut_ptr());
                    let msg = log
                        .iter()
                        .take(written as usize)
                        .map(|c| *c as u8 as char)
                        .collect::<String>();
                    return Err(VkError::Backend(format!("program link failed: {msg}")));
                }
                let mut s = self.lock()?;
                s.programs.push(prog);
                Ok((s.programs.len() - 1) as PipelineId)
            }
        }

        fn submit(&self, cmds: &[Command]) -> Result<Fence, VkError> {
            let f = &self.ctx.fns;
            let mut s = self.lock()?;
            for cmd in cmds {
                match cmd {
                    Command::Bind {
                        pipeline,
                        uniform,
                        storages,
                        image,
                    } => {
                        let prog = *s
                            .programs
                            .get(*pipeline as usize)
                            .ok_or(VkError::Backend(format!("bad pipeline id {pipeline}")))?;
                        unsafe {
                            (f.use_program)(prog);
                            if let Some(u) = uniform {
                                (f.bind_buffer_base)(
                                    sys::GL_UNIFORM_BUFFER,
                                    0,
                                    self.buf_id(&s, *u)?,
                                );
                            }
                            for (i, b) in storages.iter().enumerate() {
                                (f.bind_buffer_base)(
                                    sys::GL_SHADER_STORAGE_BUFFER,
                                    1 + i as u32,
                                    self.buf_id(&s, *b)?,
                                );
                            }
                            if let Some(img) = image {
                                let im = s
                                    .images
                                    .get(*img as usize)
                                    .ok_or(VkError::Backend(format!("bad image id {img}")))?;
                                // the storage image lives at fixed slot 7
                                (f.bind_buffer_base)(sys::GL_SHADER_STORAGE_BUFFER, 7, im.buf);
                            }
                        }
                    }
                    Command::Dispatch { groups_x, groups_y } => unsafe {
                        (f.dispatch_compute)(*groups_x, *groups_y, 1);
                    },
                    Command::Barrier => unsafe {
                        (f.memory_barrier)(
                            sys::GL_BUFFER_UPDATE_BARRIER_BIT | sys::GL_SHADER_STORAGE_BARRIER_BIT,
                        );
                    },
                    Command::CopyImageToBuffer { image, buffer } => {
                        let im = s
                            .images
                            .get(*image as usize)
                            .ok_or(VkError::Backend(format!("bad image id {image}")))?;
                        let dst = self.buf_id(&s, *buffer)?;
                        let bytes = u64::from(im.w) * u64::from(im.h) * 4;
                        unsafe {
                            (f.memory_barrier)(
                                sys::GL_BUFFER_UPDATE_BARRIER_BIT
                                    | sys::GL_SHADER_STORAGE_BARRIER_BIT,
                            );
                            // GL_COPY_READ/WRITE targets are rejected by
                            // this driver (INVALID_ENUM); SHADER_STORAGE +
                            // ARRAY_BUFFER are legal copy targets here.
                            (f.bind_buffer)(sys::GL_SHADER_STORAGE_BUFFER, im.buf);
                            (f.bind_buffer)(0x8892, dst); // GL_ARRAY_BUFFER
                            (f.copy_buffer_sub_data)(
                                sys::GL_SHADER_STORAGE_BUFFER,
                                0x8892,
                                0,
                                0,
                                bytes as sys::GLsizeiptr,
                            );
                        }
                    }
                }
            }
            unsafe {
                (f.memory_barrier)(
                    sys::GL_BUFFER_UPDATE_BARRIER_BIT | sys::GL_SHADER_STORAGE_BARRIER_BIT,
                );
                let sync = (f.fence_sync)(sys::GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
                s.fences.push(sync);
                Ok(Fence((s.fences.len() - 1) as u64))
            }
        }

        fn wait_fence(&self, fence: Fence) -> Result<(), VkError> {
            let sync = {
                let s = self.lock()?;
                *s.fences
                    .get(fence.0 as usize)
                    .ok_or(VkError::Backend(format!("bad fence id {}", fence.0)))?
            };
            let f = &self.ctx.fns;
            loop {
                let r = unsafe { (f.client_wait_sync)(sync, 0, 20_000_000) };
                if r == sys::GL_ALREADY_SIGNALED || r == sys::GL_CONDITION_SATISFIED {
                    unsafe { (f.delete_sync)(sync) };
                    return Ok(());
                }
                if r == 0x911E {
                    // WAIT_FAILED
                    return Err(VkError::Backend("client_wait_sync failed".into()));
                }
                // TIMEOUT_EXPIRED: keep waiting
            }
        }
    }
}

#[cfg(not(windows))]
mod imp_missing {
    // The GL backend is Windows-first (WGL). GLX/EGL ports follow the
    // same Device contract when needed.
}

#[cfg(test)]
mod tests {
    #[test]
    fn compiles() {}
}
