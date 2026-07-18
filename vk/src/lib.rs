#![forbid(unsafe_code)]
//! atom-vk — a minimal Vulkan-shaped compute API, composed from atoms.
//!
//! The constitution, deliberately tiny: one compute shader, one uniform
//! buffer, storage buffers, one storage image, and a copy back. Every
//! real GPU backend (wgpu today, the Atom Arc driver tomorrow) plugs in
//! behind [`Device`]. The CPU's only role here is submission — command
//! buffers and doorbells. Execution is 100% GPU.
//!
//! Vulkan vocabulary is kept on purpose: instance → device → shader
//! module → pipeline → command buffer → queue submit + fence wait.
//! When the Arc UAPI (driver Stage 6) exists, these map 1:1.

use std::fmt;

#[derive(Debug)]
pub enum VkError {
    /// This backend never supports the requested operation (named).
    Unsupported(&'static str),
    /// Mapped to an Arc driver stage that has not landed yet (named).
    StageNotReady(&'static str),
    /// A backend-specific failure string.
    Backend(String),
}

impl fmt::Display for VkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VkError::Unsupported(what) => write!(f, "unsupported on this backend: {what}"),
            VkError::StageNotReady(stage) => write!(f, "arc driver stage not ready: {stage}"),
            VkError::Backend(msg) => write!(f, "backend error: {msg}"),
        }
    }
}

impl std::error::Error for VkError {}

pub type BufferId = u32;
pub type ImageId = u32;
pub type ShaderId = u32;
pub type PipelineId = u32;

/// A shader module's source. Every backend consumes its device-native
/// form; the source text/binary IS the per-vendor ISA. `XeLp(Vec<u8>)`
/// is the Arc variant when the driver's Stage 7 compiler/runtime lands.
pub enum ShaderSource {
    /// WGSL — consumed by the wgpu backend (Vulkan/DX12/Metal).
    Wgsl(String),
    /// GLSL 430 compute — consumed by the OpenGL backend, where the
    /// VENDOR'S OWN DRIVER compiles the text (NVIDIA/AMD/Intel native).
    Glsl(String),
}

/// What a pipeline's bind point expects. The wgpu backend builds its
/// bind group layout from this; the Arc backend will build its binding
/// table from the same description.
#[derive(Clone, Copy, Debug)]
pub struct PipelineLayout {
    pub uniform: bool,
    pub storage_image: bool,
    pub storage_buffers: usize,
    /// How many TRAILING storage buffers are written by the kernel
    /// (z-buffers, accumulators). The first
    /// `storage_buffers - storage_writable` are read-only.
    pub storage_writable: usize,
}

/// One command-buffer entry. A submission is an ordered slice of these
/// — the same order a real command buffer executes.
pub enum Command {
    /// vkCmdBindPipeline + vkCmdBindDescriptorSets.
    Bind {
        pipeline: PipelineId,
        uniform: Option<BufferId>,
        storages: Vec<BufferId>,
        image: Option<ImageId>,
    },
    /// vkCmdDispatch (workgroup counts).
    Dispatch { groups_x: u32, groups_y: u32 },
    /// vkCmdCopyImageToBuffer. `read_buffer` always returns tight rows;
    /// any row-pitch padding is the backend's private business.
    CopyImageToBuffer { image: ImageId, buffer: BufferId },
    /// vkCmdPipelineBarrier: prior writes in THIS submission become
    /// visible to later commands. Multi-kernel pipelines (transform →
    /// bin → raster) ride in one submission because of this.
    Barrier,
}

/// A submission fence (vkFence). Opaque, backend-issued, cheap to
/// clone. `wait_fence` returns when the GPU has passed that point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fence(pub u64);

/// The device contract. Every method is `&self`; backends keep their
/// own interior state. All of it fails closed with a named reason.
pub trait Device {
    fn name(&self) -> &str;

    /// vkCreateBuffer (storage/uniform capable).
    fn create_buffer(&self, size: u64) -> Result<BufferId, VkError>;
    /// vkUpdateBuffer / mapped write.
    fn write_buffer(&self, buf: BufferId, offset: u64, data: &[u8]) -> Result<(), VkError>;
    /// Read the whole buffer back, tightly packed.
    fn read_buffer(&self, buf: BufferId) -> Result<Vec<u8>, VkError>;
    /// vkCreateImage: a 2-D rgba8 storage image.
    fn create_storage_image(&self, w: u32, h: u32) -> Result<ImageId, VkError>;
    /// vkCreateShaderModule.
    fn create_shader(&self, src: ShaderSource) -> Result<ShaderId, VkError>;
    /// vkCreateComputePipelines.
    fn create_pipeline(
        &self,
        shader: ShaderId,
        layout: PipelineLayout,
    ) -> Result<PipelineId, VkError>;

    /// vkQueueSubmit WITHOUT the fence wait: enqueue and return a
    /// fence immediately. This is the call that lets CPU and GPU
    /// overlap — the CPU builds the next frame while the GPU executes
    /// this one.
    fn submit(&self, cmds: &[Command]) -> Result<Fence, VkError>;

    /// vkWaitForFences: block until the GPU has passed `fence`.
    fn wait_fence(&self, fence: Fence) -> Result<(), VkError>;

    /// vkQueueSubmit + vkWaitForFences: when this returns, every
    /// command has completed on the device. The simple synchronous
    /// path — correct, but the CPU and GPU never overlap.
    fn submit_and_wait(&self, cmds: &[Command]) -> Result<(), VkError> {
        let fence = self.submit(cmds)?;
        self.wait_fence(fence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_name_their_reason() {
        assert!(VkError::Unsupported("x")
            .to_string()
            .contains("unsupported"));
        assert!(VkError::StageNotReady("stage 5")
            .to_string()
            .contains("stage 5"));
        assert!(VkError::Backend("boom".into()).to_string().contains("boom"));
    }
}
