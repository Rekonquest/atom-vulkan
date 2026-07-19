//! demo-universe — the universum substrate evolved on the GPU through
//! atom-vk. Same laws as the zero-dep CPU World (mirrored at field.rs
//! @744 lines), Jacobi ping-pong per moment, parity-checked against the
//! CPU reference compiled from the substrate crate itself.
//!
//!   cargo run -p demo-universe --release -- --bench [moments]
//!   cargo run -p demo-universe --release -- --wgpu --bench 2400
//!   cargo run -p demo-universe --release            (window, GL backend)

mod kernels;

use atom_vk::{Command, Device, PipelineLayout, ShaderSource};

const MODE_ADAPTIVE: f32 = 0.0;
const MODE_FIXED: f32 = 1.0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    dims: [f32; 4],
    rates1: [f32; 4],
    rates2: [f32; 4],
}

// ---------- SplitMix64 + init mirroring (bit-exact with the CPU World)

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut v = self.0;
        v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        v ^ (v >> 31)
    }

    fn unit_f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
}

fn observation_access(x: usize, width: usize) -> f32 {
    let nx = x as f32 / (width - 1) as f32;
    ((nx - 0.40) / 0.20).clamp(0.0, 1.0)
}

fn seed_coupling(random: &mut SplitMix64, x: usize, width: usize) -> [f32; 3] {
    let access = observation_access(x, width);
    if access <= 0.0 {
        return [0.0; 3];
    }
    let amplitude = access * (0.00018 + random.unit_f32() * 0.00062);
    [
        amplitude * (0.35 + random.unit_f32() * 0.65),
        amplitude * (0.35 + random.unit_f32() * 0.65),
        amplitude * (0.35 + random.unit_f32() * 0.65),
    ]
}

fn soft_band(distance: f32, width: f32) -> f32 {
    let normalized = (distance / width).clamp(0.0, 1.0);
    0.5 + 0.5 * (std::f32::consts::PI * normalized).cos()
}

fn edge_gate(nx: f32) -> f32 {
    ((nx - 0.18) / 0.12).clamp(0.0, 1.0)
}

fn visible_coupling(x: usize, y: usize, width: usize, height: usize) -> [f32; 3] {
    use std::f32::consts::TAU;
    let nx = x as f32 / (width - 1) as f32;
    let ny = y as f32 / (height - 1) as f32;
    let dx = (nx - 0.67) / 0.34;
    let dy = (ny - 0.50) / 0.45;
    let radius = (dx * dx + dy * dy).sqrt();
    let angle = dy.atan2(dx);
    let folded_ring = 0.52 + 0.095 * (angle * 5.0 + radius * 7.0).sin();
    let ring = soft_band((radius - folded_ring).abs(), 0.055);
    let inner = soft_band((radius - 0.19 - 0.035 * (angle * 3.0).cos()).abs(), 0.045);
    let vein = soft_band((dy - 0.19 * (nx * TAU * 2.3).sin()).abs(), 0.035)
        * soft_band((nx - 0.58).abs(), 0.42);
    let aperture = (ring.max(inner * 0.8).max(vein * 0.42) * edge_gate(nx)).clamp(0.0, 1.0);
    if aperture <= 0.001 {
        return [0.0; 3];
    }
    let hue = (angle / TAU + 1.0).fract();
    let red = (0.5 + 0.5 * (TAU * hue).cos()).powi(2);
    let green = (0.5 + 0.5 * (TAU * (hue - 1.0 / 3.0)).cos()).powi(2);
    let blue = (0.5 + 0.5 * (TAU * (hue - 2.0 / 3.0)).cos()).powi(2);
    let floor = 0.16 * aperture;
    [
        floor + aperture * red,
        floor + aperture * green,
        floor + aperture * blue,
    ]
    .map(|v| v)
}

/// The three per-site state planes: (energy.rgb, permeability),
/// (trace.rgb, activity), (coupling.rgb, phase).
type SiteVecs = (Vec<[f32; 4]>, Vec<[f32; 4]>, Vec<[f32; 4]>);

/// Initial field, bit-exact with `World::new`: per site in row-major
/// order, seed_coupling consumes 4 draws when access > 0 (adaptive /
/// inert modes), then permeability 1 draw, then phase 1 draw.
fn initial_state(mode: f32, w: usize, h: usize, seed: u64) -> SiteVecs {
    let mut random = SplitMix64(seed);
    let mut s0 = Vec::with_capacity(w * h);
    let mut s1 = Vec::with_capacity(w * h);
    let mut s2 = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let coupling = if mode == MODE_FIXED {
                visible_coupling(x, y, w, h)
            } else {
                seed_coupling(&mut random, x, w)
            };
            let permeability = 0.055 + random.unit_f32() * 0.025;
            let phase = random.unit_f32() * std::f32::consts::TAU;
            s0.push([0.0, 0.0, 0.0, permeability]);
            s1.push([0.0, 0.0, 0.0, 0.0]);
            s2.push([coupling[0], coupling[1], coupling[2], phase]);
        }
    }
    (s0, s1, s2)
}

/// `introduced` accumulation, mirroring the CPU World's per-moment loop.
fn introduced_for(moments: u64, h: usize) -> f64 {
    let mut total = 0.0f64;
    for age in 0..moments {
        let height = h as f32;
        let pulse = age as f32 * 0.037;
        let sources = [
            (0.23 + pulse.sin() * 0.025, [0.92f32, 0.08, 0.03]),
            (
                0.50 + (pulse * 0.73 + 1.2).sin() * 0.035,
                [0.05, 0.88, 0.12],
            ),
            (
                0.77 + (pulse * 0.51 + 2.4).sin() * 0.025,
                [0.04, 0.13, 0.96],
            ),
        ];
        for (position, spectrum) in sources {
            let center = (position * height).round() as isize;
            for offset in -2_isize..=2 {
                let _y = (center + offset).clamp(0, h as isize - 1);
                let envelope = (3 - offset.unsigned_abs()) as f32 / 3.0;
                let stimulus = [
                    spectrum[0] * (0.16 * envelope),
                    spectrum[1] * (0.16 * envelope),
                    spectrum[2] * (0.16 * envelope),
                ];
                total += (stimulus[0] + stimulus[1] + stimulus[2]) as f64;
            }
        }
    }
    total
}

// ---------- GPU orchestration -----------------------------------------

struct FieldWorld {
    pipeline: u32,
    uniform: u32,
    a: [u32; 3],
    b: [u32; 3],
}

fn build_world(
    dev: &impl Device,
    s0: &[[f32; 4]],
    s1: &[[f32; 4]],
    s2: &[[f32; 4]],
    gl: bool,
) -> FieldWorld {
    let src = if gl {
        ShaderSource::Glsl(kernels::step_glsl())
    } else {
        ShaderSource::Wgsl(kernels::step_wgsl())
    };
    let pipeline = dev
        .create_pipeline(
            dev.create_shader(src).expect("step shader"),
            PipelineLayout {
                uniform: true,
                storage_image: false,
                storage_buffers: 6,
                storage_writable: 3,
            },
        )
        .expect("step pipeline");
    let bytes = (s0.len() * 16) as u64;
    let mk = |data: &[[f32; 4]], label: &str| {
        let id = dev.create_buffer(bytes).expect(label);
        dev.write_buffer(id, 0, bytemuck::cast_slice(data))
            .expect("upload");
        id
    };
    let zero: Vec<[f32; 4]> = vec![[0.0; 4]; s0.len()];
    FieldWorld {
        pipeline,
        uniform: dev
            .create_buffer(std::mem::size_of::<Uniforms>() as u64)
            .expect("uniform"),
        a: [mk(s0, "a0"), mk(s1, "a1"), mk(s2, "a2")],
        b: [mk(&zero, "b0"), mk(&zero, "b1"), mk(&zero, "b2")],
    }
}

/// One moment of the simulation: which age to stamp into the uniform,
/// and which buffer pair direction to use (flip = odd age writes A).
struct Moment {
    age: u64,
    flip: bool,
}

impl Moment {
    fn at(age: u64) -> Self {
        Self {
            age,
            flip: (age % 2) == 1,
        }
    }
}

fn submit_moment(
    dev: &impl Device,
    world: &FieldWorld,
    w: usize,
    h: usize,
    mode: f32,
    rates: &[f32; 8],
    moment: Moment,
) -> atom_vk::Fence {
    let u = Uniforms {
        dims: [w as f32, h as f32, mode, moment.age as f32],
        rates1: [rates[0], rates[1], rates[2], rates[3]],
        rates2: [rates[4], rates[5], rates[6], rates[7]],
    };
    dev.write_buffer(world.uniform, 0, bytemuck::bytes_of(&u))
        .expect("write uniform");
    let (read, write) = if moment.flip {
        (world.b, world.a)
    } else {
        (world.a, world.b)
    };
    dev.submit(&[
        Command::Bind {
            pipeline: world.pipeline,
            uniform: Some(world.uniform),
            storages: vec![read[0], read[1], read[2], write[0], write[1], write[2]],
            image: None,
        },
        Command::Dispatch {
            groups_x: (w as u32).div_ceil(8),
            groups_y: (h as u32).div_ceil(8),
        },
    ])
    .expect("submit")
}

fn read_state(dev: &impl Device, world: &FieldWorld, flip: bool) -> SiteVecs {
    let read = if flip { world.b } else { world.a };
    let get = |id: u32| {
        let bytes = dev.read_buffer(id).expect("read");
        bytemuck::cast_slice::<u8, [f32; 4]>(&bytes).to_vec()
    };
    (get(read[0]), get(read[1]), get(read[2]))
}

// ---------- measurements (mirror of the CPU World's measurements) -----

struct GpuMeasurements {
    resident: f64,
    mean_permeability: f64,
    mean_coupling: f64,
    organized_fraction: f64,
    coupled_fraction: f64,
    largest_organized_component: f64,
    luminous_sites: usize,
    accounting_error: f64,
}

fn measure(
    s0: &[[f32; 4]],
    s1: &[[f32; 4]],
    s2: &[[f32; 4]],
    w: usize,
    h: usize,
    introduced: f64,
) -> GpuMeasurements {
    let mut resident = 0.0f64;
    let mut perm_sum = 0.0f64;
    let mut coup_sum = 0.0f64;
    let mut organized = 0usize;
    let mut coupled = 0usize;
    let mut luminous = 0usize;
    for i in 0..s0.len() {
        resident += (s0[i][0] + s0[i][1] + s0[i][2]) as f64;
        perm_sum += s0[i][3] as f64;
        let peak = s2[i][0].max(s2[i][1]).max(s2[i][2]);
        coup_sum += peak as f64;
        if s0[i][3] > 0.14 {
            organized += 1;
        }
        if peak > 0.012 {
            coupled += 1;
        }
        let tpeak = s1[i][0].max(s1[i][1]).max(s1[i][2]);
        if tpeak > 0.01 {
            luminous += 1;
        }
    }
    let n = s0.len() as f64;
    // largest organized component: flood fill over permeability > 0.14
    let mut visited = vec![false; s0.len()];
    let mut largest = 0usize;
    let mut stack = Vec::new();
    for start in 0..s0.len() {
        if visited[start] || s0[start][3] <= 0.14 {
            continue;
        }
        let mut size = 0usize;
        visited[start] = true;
        stack.push(start);
        while let Some(idx) = stack.pop() {
            size += 1;
            let x = idx % w;
            let y = idx / w;
            let mut visit = |j: usize| {
                if !visited[j] && s0[j][3] > 0.14 {
                    visited[j] = true;
                    stack.push(j);
                }
            };
            if x > 0 {
                visit(idx - 1);
            }
            if x + 1 < w {
                visit(idx + 1);
            }
            if y > 0 {
                visit(idx - w);
            }
            if y + 1 < h {
                visit(idx + w);
            }
        }
        largest = largest.max(size);
    }
    let radiated: f64 = s1.iter().map(|v| (v[0] + v[1] + v[2]) as f64).sum();
    // `dissipated` is a per-moment running total; it cannot be recovered
    // from final state without a reduction pass (roadmap). The honest
    // check: introduced - resident - radiated should EQUAL the CPU's
    // dissipated when parity holds.
    let partial = introduced - resident - radiated;
    GpuMeasurements {
        resident,
        mean_permeability: perm_sum / n,
        mean_coupling: coup_sum / n,
        organized_fraction: organized as f64 / n,
        coupled_fraction: coupled as f64 / n,
        largest_organized_component: largest as f64 / n,
        luminous_sites: luminous,
        accounting_error: partial,
    }
}

// ---------- probe: divergence localization ----------------------------

fn probe(dev: &impl Device, moments: u64, gl: bool) {
    let (w, h) = (192usize, 112usize);
    let mode = MODE_ADAPTIVE;
    let seed = 0xA701_5EEDu64;
    let rates = [0.105f32, 0.24, 0.022, 0.0018, 0.220, 0.00035, 0.19, 0.0012];

    let mut world = universum_substrate::World::new(universum_substrate::Config {
        width: w,
        height: h,
        seed,
        ..Default::default()
    })
    .expect("cpu world");

    let (s0, s1, s2) = initial_state(mode, w, h, seed);
    let fw = build_world(dev, &s0, &s1, &s2, gl);

    // isolate: is the initial upload even resident?
    let back = dev.read_buffer(fw.a[0]).expect("readback a0");
    let init: &[[f32; 4]] = bytemuck::cast_slice(&back);
    let mean_p = init.iter().map(|v| v[3] as f64).sum::<f64>() / init.len() as f64;
    let mean_c = init
        .iter()
        .map(|v| (v[0] + v[1] + v[2]) as f64)
        .sum::<f64>()
        / init.len() as f64;
    println!("upload check: perm {mean_p:.4} (want ~0.0675), coupling-ish {mean_c:.6}");

    // isolate: does ONE kernel step produce sane output?
    let f0 = submit_moment(dev, &fw, w, h, mode, &rates, Moment::at(0));
    dev.wait_fence(f0).expect("fence m0");
    let m0 = dev.read_buffer(fw.b[0]).expect("readback b0");
    let m0d: &[[f32; 4]] = bytemuck::cast_slice(&m0);
    let m0p = m0d.iter().map(|v| v[3] as f64).sum::<f64>() / m0d.len() as f64;
    let m0e = m0d.iter().map(|v| (v[0] + v[1] + v[2]) as f64).sum::<f64>();
    println!("after moment 0: resident {m0e:.4} (want ~1.5), perm {m0p:.4} (want ~0.0675)");
    let uback = dev.read_buffer(fw.uniform).expect("readback uniform");
    let ud: &[f32] = bytemuck::cast_slice(&uback);
    println!("uniform: {:?}", &ud[..12]);

    // seed coupling sanity, before any moments run
    let seed_mean_c = s2
        .iter()
        .map(|v| v[0].max(v[1]).max(v[2]) as f64)
        .sum::<f64>()
        / s2.len() as f64;
    println!("init coupling mean {seed_mean_c:.6} (want ~0.000205)");

    // align evolutions: moment 0 ran on the GPU above, so the CPU
    // world must take its first step here too.
    world.evolve();

    let checkpoints = [400u64, 800, 1200, 1600, 2000, 2400, 3200, 4000, 4800];
    for age in 1..=moments {
        let last = submit_moment(dev, &fw, w, h, mode, &rates, Moment::at(age));
        world.evolve();
        if checkpoints.contains(&age) {
            dev.wait_fence(last).expect("fence");
            // live buffer after moment `age`: odd ages wrote A, even wrote B
            let live = if (age % 2) == 1 { &fw.a } else { &fw.b };
            let dump = |buf: u32| -> Vec<[f32; 4]> {
                let data = dev.read_buffer(buf).expect("dump read");
                bytemuck::cast_slice::<u8, [f32; 4]>(&data).to_vec()
            };
            let d0 = dump(live[0]);
            let d1 = dump(live[1]);
            let d2 = dump(live[2]);
            let n = d0.len() as f64;
            let res: f64 = d0.iter().map(|v| (v[0] + v[1] + v[2]) as f64).sum();
            let perm: f64 = d0.iter().map(|v| v[3] as f64).sum::<f64>() / n;
            let act: f64 = d1.iter().map(|v| v[3] as f64).sum::<f64>() / n;
            let coup: f64 = d2
                .iter()
                .map(|v| v[0].max(v[1]).max(v[2]) as f64)
                .sum::<f64>()
                / n;
            let lum = d1
                .iter()
                .filter(|v| v[0].max(v[1]).max(v[2]) > 0.01)
                .count();
            let m = world.measurements();
            println!(
                "age {age:>4} gpu: res {res:>9.3} perm {perm:.4} act {act:.4} coup {coup:.6} lum {lum:>4}"
            );
            println!(
                "age {age:>4} cpu: res {:>9.3} perm {:.4}          coup {:.6} lum {:>4}",
                m.resident, m.mean_permeability, m.mean_coupling, m.luminous_sites
            );
        }
    }
}

fn bench(dev: &impl Device, moments: u64, gl: bool) {
    let (w, h) = (192usize, 112usize);
    let mode = MODE_ADAPTIVE;
    let seed = 0xA701_5EEDu64;
    let rates = [0.105f32, 0.24, 0.022, 0.0018, 0.220, 0.00035, 0.19, 0.0012];

    // CPU reference from the substrate crate itself (compiled at build).
    let cpu_t0 = std::time::Instant::now();
    let mut world = universum_substrate::World::new(universum_substrate::Config {
        width: w,
        height: h,
        seed,
        ..Default::default()
    })
    .expect("cpu world");
    world.evolve_for(moments);
    let cpu_s = cpu_t0.elapsed().as_secs_f64();
    let cpu_m = world.measurements();

    // GPU run.
    let (s0, s1, s2) = initial_state(mode, w, h, seed);
    let fw = build_world(dev, &s0, &s1, &s2, gl);
    let gpu_t0 = std::time::Instant::now();
    let mut last = submit_moment(dev, &fw, w, h, mode, &rates, Moment::at(0));
    for age in 1..moments {
        last = submit_moment(dev, &fw, w, h, mode, &rates, Moment::at(age));
        if age % 256 == 255 {
            dev.wait_fence(last).expect("checkpoint fence");
        }
    }
    dev.wait_fence(last).expect("final fence");
    let gpu_s = gpu_t0.elapsed().as_secs_f64();

    let flip = (moments % 2) == 1;
    let (g0, g1, g2) = read_state(dev, &fw, flip);
    let introduced = introduced_for(moments, h);
    let gm = measure(&g0, &g1, &g2, w, h, introduced);

    let backend = if gl { "gl" } else { "wgpu" };
    println!(
        "universum on atom-vk ({backend}) — field {w}x{h}, {moments} moments, adaptive coupling"
    );
    println!(
        "cpu: {cpu_s:.3}s ({:.0} moments/s) | gpu: {gpu_s:.3}s ({:.0} moments/s) | speedup {:.1}x",
        moments as f64 / cpu_s,
        moments as f64 / gpu_s,
        cpu_s / gpu_s
    );
    println!("{:<26} {:>14} {:>14}", "measure", "cpu", "gpu");
    let row = |name: &str, a: f64, b: f64| println!("{name:<26} {a:>14.6} {b:>14.6}");
    row("resident", cpu_m.resident, gm.resident);
    row(
        "mean_permeability",
        cpu_m.mean_permeability,
        gm.mean_permeability,
    );
    row("mean_coupling", cpu_m.mean_coupling, gm.mean_coupling);
    row(
        "organized_fraction",
        cpu_m.organized_fraction,
        gm.organized_fraction,
    );
    row(
        "coupled_fraction",
        cpu_m.coupled_fraction,
        gm.coupled_fraction,
    );
    row(
        "largest_component",
        cpu_m.largest_organized_component,
        gm.largest_organized_component,
    );
    println!(
        "{:<26} {:>14} {:>14}",
        "luminous_sites", cpu_m.luminous_sites, gm.luminous_sites
    );
    println!(
        "{:<26} {:>14.3e}",
        "cpu accounting_error", cpu_m.accounting_error
    );
    println!(
        "{:<26} {:>14.6} {:>14.6}",
        "dissipated vs gpu partial", cpu_m.dissipated, gm.accounting_error
    );

    // frame render (mirrors the substrate's rgb8) from the GPU state
    let peak_trace = g1
        .iter()
        .map(|v| v[0].max(v[1]).max(v[2]))
        .fold(0.000_001_f32, f32::max);
    let exposure = 3.2 / peak_trace.max(0.1).ln_1p();
    let mut fb = vec![0u32; w * h];
    let to_u8 =
        |value: f32| ((value.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u32) & 0xFF;
    for (i, px) in fb.iter_mut().enumerate() {
        let glow = [
            (g1[i][0].ln_1p() * exposure).min(1.0),
            (g1[i][1].ln_1p() * exposure).min(1.0),
            (g1[i][2].ln_1p() * exposure).min(1.0),
        ];
        let live = [
            (g0[i][0] * 0.30).min(1.0),
            (g0[i][1] * 0.30).min(1.0),
            (g0[i][2] * 0.30).min(1.0),
        ];
        let structure = ((g0[i][3] - 0.05) * 0.45).clamp(0.0, 0.28);
        *px = (to_u8(glow[0] + live[0] + structure * 0.38) << 16)
            | (to_u8(glow[1] + live[1] + structure * 0.55) << 8)
            | to_u8(glow[2] + live[2] + structure);
    }
    let name = format!("universe_gpu_{w}x{h}.bmp");
    let _ = atom_arena::bmp::write_bmp(std::path::Path::new(&name), w, h, &fb);
    println!("final frame: {name}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let use_wgpu = args.iter().any(|a| a == "--wgpu");
    let probe_mode = args.iter().any(|a| a == "--probe");
    let moments: u64 = args
        .iter()
        .position(|a| a == "--bench" || a == "--probe")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2400);

    if use_wgpu {
        let dev = match atom_vk_wgpu::WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no GPU: {e}");
                return;
            }
        };
        println!("device: {}", dev.name());
        if probe_mode {
            probe(&dev, moments, false);
        } else {
            bench(&dev, moments, false);
        }
    } else {
        let dev = match atom_vk_gl::GlDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no GL: {e}");
                return;
            }
        };
        println!("device: {}", dev.name());
        if probe_mode {
            probe(&dev, moments, true);
        } else {
            bench(&dev, moments, true);
        }
    }
}
