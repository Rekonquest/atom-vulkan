//! Minimal isolation probe for the GL backend: (1) buffer round-trip,
//! (2) a tiny compute shader reading one uniform float and writing one
//! pixel. Run: cargo run -p atom-vk-gl --release --example gl_probe

#[cfg(windows)]
fn main() {
    use atom_vk::{Command, Device, PipelineLayout, ShaderSource};

    let dev = match atom_vk_gl::GlDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("GL init failed: {e}");
            std::process::exit(1);
        }
    };
    println!("device: {}", dev.name());

    // 1. buffer round-trip
    let buf = dev.create_buffer(256).expect("create_buffer");
    let pattern: Vec<u8> = (0..=255u8).collect();
    dev.write_buffer(buf, 0, &pattern).expect("write_buffer");
    let back = dev.read_buffer(buf).expect("read_buffer");
    let rt_ok = back[..256] == pattern[..];
    println!("buffer round-trip: {}", if rt_ok { "PASS" } else { "FAIL" });
    println!("gl error after round-trip: {:#x}", dev.last_error());

    // 2. tiny compute: pixel = uniform float * 2 + storage[0]
    let glsl = r"#version 430
layout(local_size_x = 1) in;
layout(std140, binding = 0) uniform U { float v; } u;
layout(std430, binding = 1) readonly buffer In { float d[]; };
layout(std430, binding = 3) writeonly buffer Pixels { uint pixels[]; };
void main() {
    pixels[0] = 77u;
    pixels[1] = uint(u.v * 2.0 + d[0]);
}
";
    let shader = dev
        .create_shader(ShaderSource::Glsl(glsl.to_string()))
        .expect("create_shader");
    println!("gl error after shader: {:#x}", dev.last_error());
    let pipe = dev
        .create_pipeline(
            shader,
            PipelineLayout {
                uniform: true,
                storage_image: true,
                storage_buffers: 1,
                storage_writable: 0,
            },
        )
        .expect("create_pipeline");
    println!("gl error after pipeline: {:#x}", dev.last_error());
    let img = dev.create_storage_image(4, 4).expect("create_image");
    let ubuf = dev.create_buffer(16).expect("ubuf");
    dev.write_buffer(ubuf, 0, &21.0f32.to_le_bytes())
        .expect("write uniform");
    let sbuf = dev.create_buffer(16).expect("sbuf");
    dev.write_buffer(sbuf, 0, &1.0f32.to_le_bytes())
        .expect("write storage");
    let target = dev.create_buffer(64).expect("target");
    println!("gl error after buffers: {:#x}", dev.last_error());

    let bind = Command::Bind {
        pipeline: pipe,
        uniform: Some(ubuf),
        storages: vec![sbuf],
        image: Some(img),
    };
    let f1 = dev.submit(&[bind]).expect("submit bind");
    println!("gl error after bind-only: {:#x}", dev.last_error());
    dev.wait_fence(f1).expect("wait bind");

    let f2 = dev
        .submit(&[
            Command::Bind {
                pipeline: pipe,
                uniform: Some(ubuf),
                storages: vec![sbuf],
                image: Some(img),
            },
            Command::Dispatch {
                groups_x: 1,
                groups_y: 1,
            },
        ])
        .expect("submit dispatch");
    println!("gl error after bind+dispatch: {:#x}", dev.last_error());
    dev.wait_fence(f2).expect("wait dispatch");

    let fence = dev
        .submit(&[Command::CopyImageToBuffer {
            image: img,
            buffer: target,
        }])
        .expect("submit copy");
    println!("gl error after copy-only: {:#x}", dev.last_error());
    dev.wait_fence(fence).expect("wait copy");
    let out = dev.read_buffer(target).expect("read target");
    let px = u32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    let px1 = u32::from_le_bytes([out[4], out[5], out[6], out[7]]);
    println!("pixel[0] = {px} (expect 77 const) | pixel[1] = {px1} (expect 43 = 21*2+1)");
    println!(
        "dispatch probe: {} | uniform+storage probe: {}",
        if px == 77 { "PASS" } else { "FAIL" },
        if px1 == 43 { "PASS" } else { "FAIL" }
    );
}

#[cfg(not(windows))]
fn main() {
    eprintln!("gl_probe is Windows-only");
}
