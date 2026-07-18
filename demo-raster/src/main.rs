//! demo-raster — the graphics stage of the general Vulkan replacement:
//! triangles rasterized on the GPU through atom-vk (transform →
//! attention-bounded raster → resolve), presented through the same
//! readback window as the game. Graphics is not a pipeline object here;
//! it is compute, composed. Parity is checked against a zero-dependency
//! CPU reference rasterizer.
//!
//!   cargo run -p demo-raster --release                 (window, GL backend)
//!   cargo run -p demo-raster --release -- --wgpu
//!   cargo run -p demo-raster --release -- --bench [frames]

mod kernels;

use atom_arena::atoms::Vec2;
use atom_vk::{Command, Device, PipelineLayout, ShaderSource};

// ---------- mesh: a colored cube, 12 triangles, 4 vec4s per tri ----
// ---------- plus per-triangle uvs, 2 vec4s per tri ------------------

fn quad(
    tris: &mut Vec<[f32; 4]>,
    uvs: &mut Vec<[f32; 4]>,
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    d: [f32; 3],
    col: [f32; 3],
) {
    let (ua, ub, uc, ud) = ([0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]);
    for (v0, v1, v2, u0, u1, u2) in [(a, b, c, ua, ub, uc), (a, c, d, ua, uc, ud)] {
        tris.push([v0[0], v0[1], v0[2], 0.0]);
        tris.push([v1[0], v1[1], v1[2], 0.0]);
        tris.push([v2[0], v2[1], v2[2], 0.0]);
        tris.push([col[0], col[1], col[2], 0.0]);
        uvs.push([u0[0], u0[1], u1[0], u1[1]]);
        uvs.push([u2[0], u2[1], 0.0, 0.0]);
    }
}

fn cube() -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let c = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    let mut t = Vec::new();
    let mut u = Vec::new();
    quad(&mut t, &mut u, c[1], c[2], c[6], c[5], [1.0, 0.25, 0.2]); // +X
    quad(&mut t, &mut u, c[4], c[7], c[3], c[0], [1.0, 0.55, 0.15]); // -X
    quad(&mut t, &mut u, c[3], c[7], c[6], c[2], [0.3, 0.9, 0.4]); // +Y
    quad(&mut t, &mut u, c[4], c[5], c[1], c[0], [0.2, 0.7, 0.7]); // -Y
    quad(&mut t, &mut u, c[5], c[6], c[7], c[4], [0.3, 0.45, 1.0]); // +Z
    quad(&mut t, &mut u, c[0], c[3], c[2], c[1], [0.65, 0.35, 0.9]); // -Z
    (t, u)
}

/// The texture asset: 64×64, procedural, zero-dep. 8px checker whose
/// cells carry a red/green gradient — structure at every scale.
const TEX_SIZE: usize = 64;

fn make_texture() -> Vec<u32> {
    let mut tex = vec![0u32; TEX_SIZE * TEX_SIZE];
    for y in 0..TEX_SIZE {
        for x in 0..TEX_SIZE {
            let checker = ((x / 8 + y / 8) & 1) as f32;
            let r = x as f32 / (TEX_SIZE - 1) as f32;
            let g = y as f32 / (TEX_SIZE - 1) as f32;
            let v = 0.35 + 0.65 * checker;
            let (r, g, b) = (r * v + 0.1, g * v + 0.1, 0.9 * v);
            let q = |c: f32| ((c.clamp(0.0, 1.0) * 255.0 + 0.5) as u32) & 0xFF;
            tex[y * TEX_SIZE + x] = (q(r) << 16) | (q(g) << 8) | q(b);
        }
    }
    tex
}

// ---------- mat4, column-major (GLSL convention) --------------------

type Mat4 = [f32; 16];

fn mat_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [0.0; 16];
    for col in 0..4 {
        for row in 0..4 {
            out[col * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[col * 4 + k]).sum();
        }
    }
    out
}

fn mat_vec(m: &Mat4, v: [f32; 4]) -> [f32; 4] {
    let mut out = [0.0; 4];
    for row in 0..4 {
        out[row] = (0..4).map(|k| m[k * 4 + row] * v[k]).sum();
    }
    out
}

fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    let mut m = [0.0; 16];
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (far + near) / (near - far);
    m[11] = -1.0;
    m[14] = 2.0 * far * near / (near - far);
    m
}

fn translate(x: f32, y: f32, z: f32) -> Mat4 {
    let mut m = [0.0; 16];
    m[0] = 1.0;
    m[5] = 1.0;
    m[10] = 1.0;
    m[12] = x;
    m[13] = y;
    m[14] = z;
    m[15] = 1.0;
    m
}

fn rot_y(a: f32) -> Mat4 {
    let (s, c) = a.sin_cos();
    [
        c, 0.0, -s, 0.0, 0.0, 1.0, 0.0, 0.0, s, 0.0, c, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn rot_x(a: f32) -> Mat4 {
    let (s, c) = a.sin_cos();
    [
        1.0, 0.0, 0.0, 0.0, 0.0, c, s, 0.0, 0.0, -s, c, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn mvp(t: f32, aspect: f32) -> Mat4 {
    let p = perspective(1.05, aspect, 0.1, 100.0);
    let v = translate(0.0, 0.0, -4.5);
    let r = mat_mul(&rot_y(t), &rot_x(t * 0.6));
    mat_mul(&p, &mat_mul(&v, &r))
}

// ---------- the uniform, matching `struct U` in both kernels --------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    mvp: [f32; 16],
    params: [f32; 4],
}

// ---------- zero-dep CPU reference rasterizer -----------------------

fn tex_sample(tex: &[u32], uv: (f32, f32)) -> (f32, f32, f32) {
    let fr = |v: f32| v - v.floor();
    let (u, v) = (fr(uv.0), fr(uv.1));
    let (tx, ty) = (u * TEX_SIZE as f32 - 0.5, v * TEX_SIZE as f32 - 0.5);
    let (bx, by) = (tx.floor() as i32, ty.floor() as i32);
    let (fx, fy) = (tx - bx as f32, ty - by as f32);
    let sz = TEX_SIZE as i32;
    let get = |x: i32, y: i32| {
        let v = tex[y.rem_euclid(sz) as usize * TEX_SIZE + x.rem_euclid(sz) as usize];
        (
            ((v >> 16) & 0xFF) as f32 / 255.0,
            ((v >> 8) & 0xFF) as f32 / 255.0,
            (v & 0xFF) as f32 / 255.0,
        )
    };
    let c00 = get(bx, by);
    let c10 = get(bx + 1, by);
    let c01 = get(bx, by + 1);
    let c11 = get(bx + 1, by + 1);
    let lp = |a: (f32, f32, f32), b: (f32, f32, f32), t: f32| {
        (
            a.0 + (b.0 - a.0) * t,
            a.1 + (b.1 - a.1) * t,
            a.2 + (b.2 - a.2) * t,
        )
    };
    lp(lp(c00, c10, fx), lp(c01, c11, fx), fy)
}

fn cpu_frame(
    tris: &[[f32; 4]],
    uvs: &[[f32; 4]],
    tex: &[u32],
    m: &Mat4,
    w: usize,
    h: usize,
) -> Vec<u32> {
    let n_tris = tris.len() / 4;
    let mut zbuf = vec![u32::MAX; w * h];
    let mut aux = vec![0.0f32; n_tris];
    let edge = |a: Vec2, b: Vec2, p: Vec2| (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);

    for tri in 0..n_tris {
        let (w0, w1, w2) = (tris[tri * 4], tris[tri * 4 + 1], tris[tri * 4 + 2]);
        let n = {
            let (u, v) = (
                [w1[0] - w0[0], w1[1] - w0[1], w1[2] - w0[2]],
                [w2[0] - w0[0], w2[1] - w0[1], w2[2] - w0[2]],
            );
            let c = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
            if l > 0.0 {
                [c[0] / l, c[1] / l, c[2] / l]
            } else {
                [0.0, 0.0, 1.0]
            }
        };
        let l = {
            let v = [0.5f32, 0.8, -0.6];
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / len, v[1] / len, v[2] / len]
        };
        aux[tri] = 0.25 + 0.75 * (n[0] * l[0] + n[1] * l[1] + n[2] * l[2]).max(0.0);

        let c0 = mat_vec(m, [w0[0], w0[1], w0[2], 1.0]);
        let c1 = mat_vec(m, [w1[0], w1[1], w1[2], 1.0]);
        let c2 = mat_vec(m, [w2[0], w2[1], w2[2], 1.0]);
        if c0[3] <= 0.0 || c1[3] <= 0.0 || c2[3] <= 0.0 {
            continue;
        }
        let (wf, hf) = (w as f32, h as f32);
        let s = |c: [f32; 4]| {
            (
                Vec2::new(
                    (c[0] / c[3] * 0.5 + 0.5) * wf,
                    (0.5 - c[1] / c[3] * 0.5) * hf,
                ),
                c[2] / c[3] * 0.5 + 0.5,
            )
        };
        let (s0, z0) = s(c0);
        let (s1, z1) = s(c1);
        let (s2, z2) = s(c2);

        let x0 = s0.x.min(s1.x).min(s2.x).floor().max(0.0) as usize;
        let y0 = s0.y.min(s1.y).min(s2.y).floor().max(0.0) as usize;
        let x1 = (s0.x.max(s1.x).max(s2.x).ceil() as usize).min(w - 1);
        let y1 = (s0.y.max(s1.y).max(s2.y).ceil() as usize).min(h - 1);

        for y in y0..=y1 {
            for x in x0..=x1 {
                let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let (e0, e1, e2) = (edge(s1, s2, p), edge(s2, s0, p), edge(s0, s1, p));
                let pos = e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0;
                let neg = e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0;
                if !pos && !neg {
                    continue;
                }
                let area = e0.abs() + e1.abs() + e2.abs();
                if area < 1e-9 {
                    continue;
                }
                let z = (e0.abs() * z0 + e1.abs() * z1 + e2.abs() * z2) / area;
                let packed = (z.to_bits() & 0xFFFF_FF00) | (tri as u32 & 0xFF);
                let idx = y * w + x;
                if packed < zbuf[idx] {
                    zbuf[idx] = packed;
                }
            }
        }
    }

    let mut fb = vec![0u32; w * h];
    let edge2 = |a: Vec2, b: Vec2, p: Vec2| (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
    for (i, packed) in zbuf.iter().enumerate() {
        let (r, g, b) = if *packed != u32::MAX {
            let tri = (*packed & 0xFF) as usize;
            let c = tris[tri * 4 + 3];
            let sh = aux[tri];
            let (w0, w1, w2) = (tris[tri * 4], tris[tri * 4 + 1], tris[tri * 4 + 2]);
            let c0 = mat_vec(m, [w0[0], w0[1], w0[2], 1.0]);
            let c1 = mat_vec(m, [w1[0], w1[1], w1[2], 1.0]);
            let c2 = mat_vec(m, [w2[0], w2[1], w2[2], 1.0]);
            let (wf, hf) = (w as f32, h as f32);
            let sp = |c: [f32; 4]| {
                Vec2::new(
                    (c[0] / c[3] * 0.5 + 0.5) * wf,
                    (0.5 - c[1] / c[3] * 0.5) * hf,
                )
            };
            let (s0, s1, s2) = (sp(c0), sp(c1), sp(c2));
            let p = Vec2::new((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
            let (e0, e1, e2) = (
                edge2(s1, s2, p).abs(),
                edge2(s2, s0, p).abs(),
                edge2(s0, s1, p).abs(),
            );
            let area = e0 + e1 + e2;
            let uv = (
                (e0 * uvs[tri * 2][0] + e1 * uvs[tri * 2][2] + e2 * uvs[tri * 2 + 1][0]) / area,
                (e0 * uvs[tri * 2][1] + e1 * uvs[tri * 2][3] + e2 * uvs[tri * 2 + 1][1]) / area,
            );
            let t = tex_sample(tex, uv);
            (c[0] * t.0 * sh, c[1] * t.1 * sh, c[2] * t.2 * sh)
        } else {
            (0.05, 0.06, 0.09)
        };
        let q = |v: f32| ((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32) & 0xFF;
        fb[i] = (q(r) << 16) | (q(g) << 8) | q(b);
    }
    fb
}

// ---------- GPU orchestration ----------------------------------------

struct RastWorld {
    pipe_fuse: u32,
    uniform: u32,
    tris: u32,
    uvs: u32,
    tex: u32,
    image: u32,
    targets: [u32; 2],
}

fn build_world(
    dev: &impl Device,
    tris: &[[f32; 4]],
    uvs: &[[f32; 4]],
    tex: &[u32],
    w: u32,
    h: u32,
    gl: bool,
) -> RastWorld {
    let fuse_src = if gl {
        ShaderSource::Glsl(kernels::fuse_glsl())
    } else {
        ShaderSource::Wgsl(kernels::fuse_wgsl())
    };
    let pipe_fuse = dev
        .create_pipeline(
            dev.create_shader(fuse_src).expect("fuse shader"),
            PipelineLayout {
                uniform: true,
                storage_image: true,
                storage_buffers: 3,
                storage_writable: 0,
            },
        )
        .expect("fuse pipeline");

    let frame_bytes = u64::from(w) * u64::from(h) * 4;
    let world = RastWorld {
        pipe_fuse,
        uniform: dev
            .create_buffer(std::mem::size_of::<Uniforms>() as u64)
            .expect("uniform"),
        tris: dev.create_buffer((tris.len() * 16) as u64).expect("tris"),
        uvs: dev.create_buffer((uvs.len() * 16) as u64).expect("uvs"),
        tex: dev.create_buffer((tex.len() * 4) as u64).expect("tex"),
        image: dev.create_storage_image(w, h).expect("image"),
        targets: [
            dev.create_buffer(frame_bytes).expect("target A"),
            dev.create_buffer(frame_bytes).expect("target B"),
        ],
    };
    dev.write_buffer(world.tris, 0, bytemuck::cast_slice(tris))
        .expect("upload tris");
    dev.write_buffer(world.uvs, 0, bytemuck::cast_slice(uvs))
        .expect("upload uvs");
    dev.write_buffer(world.tex, 0, bytemuck::cast_slice(tex))
        .expect("upload tex");
    world
}

/// One frame: one fused dispatch + one copy, one submission, 80 bytes
/// of upload (the uniform). The graphics pipeline as compute.
fn submit_frame(
    dev: &impl Device,
    world: &RastWorld,
    m: &Mat4,
    w: u32,
    h: u32,
    n_tris: u32,
    target: u32,
) -> atom_vk::Fence {
    let u = Uniforms {
        mvp: *m,
        params: [n_tris as f32, w as f32, h as f32, TEX_SIZE as f32],
    };
    dev.write_buffer(world.uniform, 0, bytemuck::bytes_of(&u))
        .expect("write uniform");
    dev.submit(&[
        Command::Bind {
            pipeline: world.pipe_fuse,
            uniform: Some(world.uniform),
            storages: vec![world.tris, world.uvs, world.tex],
            image: Some(world.image),
        },
        Command::Dispatch {
            groups_x: w.div_ceil(8),
            groups_y: h.div_ceil(8),
        },
        Command::CopyImageToBuffer {
            image: world.image,
            buffer: target,
        },
    ])
    .expect("submit")
}

fn parity_check(dev: &impl Device, tris: &[[f32; 4]], uvs: &[[f32; 4]], tex: &[u32], gl: bool) {
    let (w, h) = (480u32, 270u32);
    let world = build_world(dev, tris, uvs, tex, w, h, gl);
    let m = mvp(0.7, w as f32 / h as f32);
    let fence = submit_frame(
        dev,
        &world,
        &m,
        w,
        h,
        (tris.len() / 4) as u32,
        world.targets[0],
    );
    dev.wait_fence(fence).expect("wait");
    let bytes = dev.read_buffer(world.targets[0]).expect("read");
    let gpu_fb: &[u32] = bytemuck::cast_slice(&bytes);
    let cpu_fb = cpu_frame(tris, uvs, tex, &m, w as usize, h as usize);

    let mut mism = 0usize;
    let mut maxd = 0i32;
    for (a, b) in cpu_fb.iter().zip(gpu_fb.iter()) {
        let mut worst = 0i32;
        for ch in 0..3 {
            let d = (((a >> (ch * 8)) & 0xff) as i32 - ((b >> (ch * 8)) & 0xff) as i32).abs();
            worst = worst.max(d);
        }
        maxd = maxd.max(worst);
        if worst > 2 {
            mism += 1;
        }
    }
    let total = cpu_fb.len();
    let pct = 100.0 * (total - mism) as f64 / total as f64;
    let verdict = if pct >= 99.5 { "PASS" } else { "FAIL" };
    println!(
        "parity (gpu raster vs zero-dep cpu raster, 480x270): {pct:.2}% pixels within 2 LSB | max diff {maxd} | {verdict}"
    );
}

fn bench(dev: &impl Device, frames: usize, gl: bool) {
    let (tris, uvs) = cube();
    let tex = make_texture();
    let n_tris = (tris.len() / 4) as u32;
    parity_check(dev, &tris, &uvs, &tex, gl);
    println!("frames per leg: {frames}");
    println!(
        "{:>11} | {:>18} | {:>14} | {:>9}",
        "resolution", "vk raster full frame", "cpu raster", "speedup"
    );
    let mut last: Option<(Vec<u32>, u32, u32)> = None;
    for &(w, h) in &[(480u32, 270u32), (960, 540), (1920, 1080)] {
        let world = build_world(dev, &tris, &uvs, &tex, w, h, gl);
        let t0 = std::time::Instant::now();
        for f in 0..frames {
            let m = mvp(f as f32 / 60.0, w as f32 / h as f32);
            submit_frame(dev, &world, &m, w, h, n_tris, world.targets[f % 2]);
            if f > 0 {
                let _ = dev.read_buffer(world.targets[(f - 1) % 2]).expect("read");
            }
        }
        let bytes = dev
            .read_buffer(world.targets[(frames - 1) % 2])
            .expect("read");
        let fb = bytemuck::cast_slice(&bytes).to_vec();
        let gpu_s = t0.elapsed().as_secs_f64();

        let cpu_frames = if u64::from(w) * u64::from(h) > 4_000_000 {
            60
        } else {
            frames.min(300)
        };
        let t0 = std::time::Instant::now();
        for f in 0..cpu_frames {
            let m = mvp(f as f32 / 60.0, w as f32 / h as f32);
            let _ = cpu_frame(&tris, &uvs, &tex, &m, w as usize, h as usize);
        }
        let cpu_s = t0.elapsed().as_secs_f64();

        let gpu_fps = frames as f64 / gpu_s;
        let cpu_fps = cpu_frames as f64 / cpu_s;
        println!(
            "{w:>5}x{h:<5} | {:>15.1} fps | {:>11.1} fps | {:>8.1}x",
            gpu_fps,
            cpu_fps,
            gpu_fps / cpu_fps
        );
        if (w, h) == (1920, 1080) {
            last = Some((fb, w, h));
        }
    }
    if let Some((fb, w, h)) = last {
        let commitment = atom_arena::atoms::hash_bytes(bytemuck::cast_slice(&fb));
        let name = format!("raster_bench_{w}x{h}.bmp");
        let _ =
            atom_arena::bmp::write_bmp(std::path::Path::new(&name), w as usize, h as usize, &fb);
        println!("final frame: {name} | commitment tag: {commitment:#018x}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let use_wgpu = args.iter().any(|a| a == "--wgpu");
    let bench_frames: Option<usize> = args
        .iter()
        .position(|a| a == "--bench")
        .map(|i| args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(600));

    if use_wgpu {
        let dev = match atom_vk_wgpu::WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no GPU: {e}");
                return;
            }
        };
        println!("device: {}", dev.name());
        run(&dev, bench_frames, false);
    } else {
        let dev = match atom_vk_gl::GlDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no GL: {e}");
                return;
            }
        };
        println!("device: {}", dev.name());
        run(&dev, bench_frames, true);
    }
}

fn run(dev: &impl Device, bench_frames: Option<usize>, gl: bool) {
    if let Some(frames) = bench_frames {
        bench(dev, frames, gl);
        return;
    }
    #[cfg(windows)]
    run_window(dev, gl);
    #[cfg(not(windows))]
    bench(dev, 600, gl);
}

#[cfg(windows)]
fn run_window(dev: &impl Device, gl: bool) {
    use atom_arena::win32::Window;

    let (tris, uvs) = cube();
    let tex = make_texture();
    let n_tris = (tris.len() / 4) as u32;
    let (w, h) = (480u32, 270u32);
    let world = build_world(dev, &tris, &uvs, &tex, w, h, gl);
    let mut win = match Window::new("ATOM-VK RASTER", (w * 2) as usize, (h * 2) as usize) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("window failed: {e}");
            return;
        }
    };

    let start = Window::time();
    let mut fps_t = start;
    let mut fps_frames = 0usize;
    let mut frame_no = 0usize;
    while win.poll() {
        let now = Window::time();
        if win.key(0x1B) {
            break;
        }
        let m = mvp((now - start) as f32, w as f32 / h as f32);
        submit_frame(dev, &world, &m, w, h, n_tris, world.targets[frame_no % 2]);
        if frame_no > 0 {
            let bytes = dev
                .read_buffer(world.targets[(frame_no - 1) % 2])
                .expect("read");
            let fb: &[u32] = bytemuck::cast_slice(&bytes);
            win.present(fb, w as usize, h as usize);
        }
        frame_no += 1;

        fps_frames += 1;
        if now - fps_t >= 0.5 {
            let fps = fps_frames as f64 / (now - fps_t);
            fps_frames = 0;
            fps_t = now;
            win.set_title(&format!(
                "ATOM-VK RASTER — {fps:.0} fps — graphics as compute"
            ));
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}
