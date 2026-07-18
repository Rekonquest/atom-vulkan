#![forbid(unsafe_code)]
//! atom-vk-arc — the Atom Arc driver backend: a walking skeleton.
//!
//! Every API operation names the Atom-Intel-Driver stage that will
//! implement it (see `docs/STACKING_ORDER.md` in Atom-Intel-Driver).
//! Until that stage lands, the operation fails closed with
//! [`VkError::StageNotReady`] — never silently, never partially.
//!
//! Mapping (driver stages):
//!   4. Memory foundations — LMEM allocator + GGTT PTE map/pin/flush
//!      (DRIVER STATE: landed through Stage 4V)
//!   5. Execution foundations — command queues, doorbells, fences
//!      (DRIVER STATE: not started → submissions fail closed here)
//!   6. Render/compute interface — headless DRM render node + small
//!      versioned UAPI (the syscall boundary this crate will call)
//!   7. User-space runtime — validation + Xe-LP compiler/runtime
//!      (shader modules become real here)

use atom_vk::{BufferId, ImageId, PipelineId, ShaderId};
use atom_vk::{Command, Device, PipelineLayout, ShaderSource, VkError};

/// The earliest driver stage whose absence blocks each operation.
const STAGE_5_EXEC: &str = "stage 5 execution foundations (command queues/doorbells/fences)";
const STAGE_6_UAPI: &str = "stage 6 render/compute UAPI (DRM render node)";
const STAGE_7_RUNTIME: &str = "stage 7 user-space runtime (Xe-LP compiler)";

/// The API-op → driver-stage map, as data, so tooling and docs can
/// print it without reading source.
pub fn stage_map() -> &'static [(&'static str, &'static str)] {
    &[
        ("create_buffer", STAGE_6_UAPI),
        ("write_buffer", STAGE_6_UAPI),
        ("read_buffer", STAGE_6_UAPI),
        ("create_storage_image", STAGE_6_UAPI),
        ("create_shader", STAGE_7_RUNTIME),
        ("create_pipeline", STAGE_7_RUNTIME),
        ("submit", STAGE_5_EXEC),
        ("wait_fence", STAGE_5_EXEC),
        ("barrier (command)", STAGE_5_EXEC),
    ]
}

/// The skeleton device. Constructing it is free; operating it names
/// exactly what the driver still owes.
pub struct ArcDevice {
    name: String,
}

impl ArcDevice {
    pub fn new() -> ArcDevice {
        ArcDevice {
            name: "atom arc (a380, driver stages 1-4v landed)".to_string(),
        }
    }
}

impl Default for ArcDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl Device for ArcDevice {
    fn name(&self) -> &str {
        &self.name
    }

    fn create_buffer(&self, _size: u64) -> Result<BufferId, VkError> {
        // Memory foundations (Stage 4) exist in-kernel, but there is no
        // userspace syscall boundary to allocate through yet.
        Err(VkError::StageNotReady(STAGE_6_UAPI))
    }

    fn write_buffer(&self, _buf: BufferId, _offset: u64, _data: &[u8]) -> Result<(), VkError> {
        Err(VkError::StageNotReady(STAGE_6_UAPI))
    }

    fn read_buffer(&self, _buf: BufferId) -> Result<Vec<u8>, VkError> {
        Err(VkError::StageNotReady(STAGE_6_UAPI))
    }

    fn create_storage_image(&self, _w: u32, _h: u32) -> Result<ImageId, VkError> {
        Err(VkError::StageNotReady(STAGE_6_UAPI))
    }

    fn create_shader(&self, _src: ShaderSource) -> Result<ShaderId, VkError> {
        // A real shader module is a Xe-LP binary. The driver needs its
        // compiler/runtime integration before this can accept anything.
        Err(VkError::StageNotReady(STAGE_7_RUNTIME))
    }

    fn create_pipeline(
        &self,
        _shader: ShaderId,
        _layout: PipelineLayout,
    ) -> Result<PipelineId, VkError> {
        Err(VkError::StageNotReady(STAGE_7_RUNTIME))
    }

    fn submit(&self, _cmds: &[Command]) -> Result<atom_vk::Fence, VkError> {
        // Command queues, doorbells, and fences are the next driver
        // layer after memory foundations.
        Err(VkError::StageNotReady(STAGE_5_EXEC))
    }

    fn wait_fence(&self, _fence: atom_vk::Fence) -> Result<(), VkError> {
        Err(VkError::StageNotReady(STAGE_5_EXEC))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_operation_fails_closed_with_a_named_stage() {
        let dev = ArcDevice::new();
        assert!(matches!(
            dev.create_buffer(16),
            Err(VkError::StageNotReady(s)) if s.contains("stage 6")
        ));
        assert!(matches!(
            dev.submit_and_wait(&[]),
            Err(VkError::StageNotReady(s)) if s.contains("stage 5")
        ));
        assert!(matches!(
            dev.create_shader(ShaderSource::Wgsl(String::new())),
            Err(VkError::StageNotReady(s)) if s.contains("stage 7")
        ));
    }

    #[test]
    fn stage_map_covers_the_whole_api() {
        assert_eq!(stage_map().len(), 9);
    }
}
