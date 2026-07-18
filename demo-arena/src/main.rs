//! demo-arena — the atom-arena game driven entirely through the atom-vk
//! API. The same Bind/Dispatch/Copy command stream any Vulkan client
//! emits; parity is checked against the game's own zero-dependency CPU
//! renderer, which lives outside the API. The Arc walking skeleton
//! reports which driver stages each op is waiting on.
//!
//!   cargo run --release -p demo-arena -- --bench [frames] [WxH]
//!   cargo run --release -p demo-arena            (playable window)

use atom_arena::atoms::Vec2;
use atom_arena::orchestrator::{default_policy, Game, Input, Policy};
use atom_vk::{Command, Device, PipelineLayout, ShaderSource};

fn bench_policy() -> Policy {
    let mut pol = default_policy();
    pol.spawn_interval = 0.35;
    pol.spawn_interval_decay = 0.97;
    pol.spawn_interval_min = 0.15;
    pol
}

/// The same scripted pilot every other harness uses — deterministic.
fn scripted_input(game: &Game, f: usize) -> Input {
    let t = f as f32 / 60.0;
    match game.nearest_enemy(game.player_pos()) {
        Some(target) => {
            let to = target.sub(game.player_pos());
            let away = if atom_arena::atoms::length(to) < 4.0 {
                -1.0
            } else {
                1.0
            };
            Input {
                mv: Vec2::new(
                    to.y.signum() * 0.8 + to.x * 0.1 * away,
                    -to.x.signum() * 0.8 + to.y * 0.1 * away,
                ),
                aim: to,
                fire: true,
                restart: true,
            }
        }
        None => Input {
            mv: Vec2::new(t.cos(), (t * 0.7).sin()),
            aim: Vec2::new((t * 1.7).cos(), (t * 1.7).sin()),
            fire: true,
            restart: true,
        },
    }
}

/// Everything the API needs for one resolution, created once.
struct VkWorld {
    pipeline: u32,
    image: u32,
    uniform: u32,
    discs: u32,
    segs: u32,
    targets: [u32; 2],
}

fn build_world(dev: &impl Device, pol: &Policy, game: &Game, w: u32, h: u32, gl: bool) -> VkWorld {
    let snap = game.snapshot();
    let src = if gl {
        ShaderSource::Glsl(atom_arena_gpu::glsl::build_shader(snap.shapes))
    } else {
        ShaderSource::Wgsl(atom_arena_gpu::wgsl::build_shader(snap.shapes))
    };
    let shader = dev.create_shader(src).expect("create_shader");
    let pipeline = dev
        .create_pipeline(
            shader,
            PipelineLayout {
                uniform: true,
                storage_image: true,
                storage_buffers: 2,
                storage_writable: 0,
            },
        )
        .expect("create_pipeline");
    let frame_bytes = u64::from(w) * u64::from(h) * 4;
    VkWorld {
        pipeline,
        image: dev
            .create_storage_image(w, h)
            .expect("create_storage_image"),
        uniform: dev
            .create_buffer(std::mem::size_of::<atom_arena_gpu::Uniforms>() as u64)
            .expect("uniform buffer"),
        discs: dev
            .create_buffer(((pol.enemy_cap + 2) * 16) as u64)
            .expect("discs buffer"),
        segs: dev
            .create_buffer((atom_arena_gpu::MAX_SEGS * 2 * 16) as u64)
            .expect("segs buffer"),
        targets: [
            dev.create_buffer(frame_bytes).expect("target A"),
            dev.create_buffer(frame_bytes).expect("target B"),
        ],
    }
}

/// Upload one frame of game state and submit its commands WITHOUT
/// waiting — the fence is the caller's to hold. `copy_target` selects
/// the readback buffer the image lands in; None means dispatch only.
fn submit_frame(
    dev: &impl Device,
    world: &VkWorld,
    pol: &Policy,
    game: &Game,
    w: u32,
    h: u32,
    copy_target: Option<u32>,
) -> atom_vk::Fence {
    let snap = game.snapshot();
    let (u, discs, segs) = atom_arena_gpu::build_frame(pol, &snap, w, h);
    dev.write_buffer(world.uniform, 0, bytemuck::bytes_of(&u))
        .expect("write uniform");
    dev.write_buffer(world.discs, 0, bytemuck::cast_slice(&discs))
        .expect("write discs");
    dev.write_buffer(world.segs, 0, bytemuck::cast_slice(&segs))
        .expect("write segs");
    let mut cmds = vec![
        Command::Bind {
            pipeline: world.pipeline,
            uniform: Some(world.uniform),
            storages: vec![world.discs, world.segs],
            image: Some(world.image),
        },
        Command::Dispatch {
            groups_x: w.div_ceil(8),
            groups_y: h.div_ceil(8),
        },
    ];
    if let Some(target) = copy_target {
        cmds.push(Command::CopyImageToBuffer {
            image: world.image,
            buffer: target,
        });
    }
    dev.submit(&cmds).expect("submit")
}

fn read_target(dev: &impl Device, target: u32) -> Vec<u32> {
    let bytes = dev.read_buffer(target).expect("read_buffer");
    bytemuck::cast_slice(&bytes).to_vec()
}

/// Synchronous single-frame path — used by the parity gate, where the
/// frame read back must be exactly the frame submitted.
fn frame_through_api_sync(
    dev: &impl Device,
    world: &VkWorld,
    pol: &Policy,
    game: &Game,
    w: u32,
    h: u32,
) -> Vec<u32> {
    let fence = submit_frame(dev, world, pol, game, w, h, Some(world.targets[0]));
    dev.wait_fence(fence).expect("wait_fence");
    read_target(dev, world.targets[0])
}

fn parity_check(dev: &impl Device, pol: &Policy, gl: bool) {
    let (w, h) = (480u32, 270u32);
    let mut game = Game::new(*pol);
    for f in 0..300 {
        game.step(&scripted_input(&game, f), 1.0 / 60.0);
    }
    let world = build_world(dev, pol, &game, w, h, gl);
    let gpu_fb = frame_through_api_sync(dev, &world, pol, &game, w, h);
    let mut cpu_fb = vec![0u32; (w * h) as usize];
    game.render_world(&mut cpu_fb, w as usize, h as usize);

    let mut maxd = 0i32;
    let mut sum = 0f64;
    for (a, b) in cpu_fb.iter().zip(gpu_fb.iter()) {
        for ch in 0..3 {
            let d = (((a >> (ch * 8)) & 0xff) as i32 - ((b >> (ch * 8)) & 0xff) as i32).abs();
            maxd = maxd.max(d);
            sum += d as f64;
        }
    }
    let verdict = if maxd <= 2 { "PASS" } else { "FAIL" };
    println!(
        "parity (atom-vk vs zero-dep cpu renderer, 480x270 frame 300): max diff {maxd} | mean {:.3} | {verdict}",
        sum / (cpu_fb.len() * 3) as f64
    );
}

fn bench(dev: &impl Device, frames: usize, custom: Option<(u32, u32)>, gl: bool) {
    let pol = bench_policy();
    println!("frames per leg: {frames} (cpu legs capped)");
    parity_check(dev, &pol, gl);

    let resolutions: Vec<(u32, u32)> = match custom {
        Some(r) => vec![r],
        None => vec![(480, 270), (960, 540), (1920, 1080)],
    };
    println!(
        "{:>11} | {:>16} | {:>16} | {:>12} | {:>9}",
        "resolution", "vk submit-only", "vk full frame", "cpu render", "speedup"
    );
    let mut last: Option<(Vec<u32>, u32, u32)> = None;
    for &(w, h) in &resolutions {
        let mut game = Game::new(pol);
        let world = build_world(dev, &pol, &game, w, h, gl);

        let t0 = std::time::Instant::now();
        let mut last_pure = None;
        for f in 0..frames {
            game.step(&scripted_input(&game, f), 1.0 / 60.0);
            last_pure = Some(submit_frame(dev, &world, &pol, &game, w, h, None));
            if f % 64 == 63 {
                dev.wait_fence(last_pure.unwrap()).expect("pure fence");
            }
        }
        if let Some(f) = last_pure {
            dev.wait_fence(f).expect("pure fence");
        }
        let pure_s = t0.elapsed().as_secs_f64();

        // Pipelined full frame: double-buffered targets; read frame N-1
        // while frame N is in flight. CPU and GPU overlap — the fence
        // is what makes the overlap honest.
        let t0 = std::time::Instant::now();
        let mut fb = Vec::new();
        for f in 0..frames {
            game.step(&scripted_input(&game, f), 1.0 / 60.0);
            submit_frame(dev, &world, &pol, &game, w, h, Some(world.targets[f % 2]));
            if f > 0 {
                fb = read_target(dev, world.targets[(f - 1) % 2]);
            }
        }
        if frames > 0 {
            fb = read_target(dev, world.targets[(frames - 1) % 2]);
        }
        let full_s = t0.elapsed().as_secs_f64();

        let cpu_frames = if u64::from(w) * u64::from(h) > 4_000_000 {
            60
        } else {
            frames.min(300)
        };
        let mut cpu_fb = vec![0u32; (w * h) as usize];
        let t0 = std::time::Instant::now();
        for f in 0..cpu_frames {
            game.step(&scripted_input(&game, f), 1.0 / 60.0);
            game.render_world(&mut cpu_fb, w as usize, h as usize);
        }
        let cpu_s = t0.elapsed().as_secs_f64();

        let pure_fps = frames as f64 / pure_s;
        let full_fps = frames as f64 / full_s;
        let cpu_fps = cpu_frames as f64 / cpu_s;
        println!(
            "{w:>5}x{h:<5} | {:>13.1} fps | {:>13.1} fps | {:>9.1} fps | {:>8.1}x",
            pure_fps,
            full_fps,
            cpu_fps,
            full_fps / cpu_fps
        );
        if (w, h) == *resolutions.last().unwrap() {
            game.render_hud(&mut fb, w as usize, h as usize);
            last = Some((fb, w, h));
        }
    }

    if let Some((fb, w, h)) = last {
        let name = format!("atomvk_bench_{w}x{h}.bmp");
        match atom_arena::bmp::write_bmp(std::path::Path::new(&name), w as usize, h as usize, &fb) {
            Ok(()) => println!("final frame: {name}"),
            Err(e) => println!("bmp write failed: {e}"),
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let arc = atom_vk_arc::ArcDevice::new();
    println!("arc backend: {}", arc.name());
    for (op, stage) in atom_vk_arc::stage_map() {
        println!("  {op:20} -> {stage}");
    }

    let gl = args.iter().any(|a| a == "--gl");
    if gl {
        let dev = match atom_vk_gl::GlDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no GL: {e}");
                std::process::exit(1);
            }
        };
        println!("device: {}", dev.name());
        run(&dev, &args, true);
    } else {
        let dev = match atom_vk_wgpu::WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no GPU: {e}");
                std::process::exit(1);
            }
        };
        println!("device: {}", dev.name());
        run(&dev, &args, false);
    }
}

fn run(dev: &impl Device, args: &[String], gl: bool) {
    if let Some(i) = args.iter().position(|a| a == "--bench") {
        let frames = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(600);
        let custom = args.get(i + 2).and_then(|s| {
            let (w, h) = s.split_once('x')?;
            Some((w.parse().ok()?, h.parse().ok()?))
        });
        bench(dev, frames, custom, gl);
        return;
    }

    #[cfg(windows)]
    run_window(dev, gl);
    #[cfg(not(windows))]
    bench(dev, 600, None, gl);
}

#[cfg(windows)]
fn run_window(dev: &impl Device, gl: bool) {
    use atom_arena::win32::Window;

    const VK_LBUTTON: i32 = 0x01;
    const VK_ESCAPE: i32 = 0x1B;
    const VK_SPACE: i32 = 0x20;
    const VK_LEFT: i32 = 0x25;
    const VK_UP: i32 = 0x26;
    const VK_RIGHT: i32 = 0x27;
    const VK_DOWN: i32 = 0x28;

    let pol = default_policy();
    let upscale = 2;
    let (w, h) = (pol.render_w as u32, pol.render_h as u32);
    let mut game = Game::new(pol);
    let world = build_world(dev, &pol, &game, w, h, gl);
    let mut win = match Window::new(
        "ATOM ARENA — atom-vk",
        pol.render_w * upscale,
        pol.render_h * upscale,
    ) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("window failed: {e}");
            return;
        }
    };

    let mut last = Window::time();
    let mut fps_t = last;
    let mut fps_frames = 0usize;
    let mut frame_no = 0usize;
    while win.poll() {
        let now = Window::time();
        let dt = (now - last) as f32;
        last = now;

        let mut mv = Vec2::ZERO;
        if win.key(b'W' as i32) || win.key(VK_UP) {
            mv.y += 1.0;
        }
        if win.key(b'S' as i32) || win.key(VK_DOWN) {
            mv.y -= 1.0;
        }
        if win.key(b'A' as i32) || win.key(VK_LEFT) {
            mv.x -= 1.0;
        }
        if win.key(b'D' as i32) || win.key(VK_RIGHT) {
            mv.x += 1.0;
        }
        if win.key(VK_ESCAPE) {
            break;
        }

        let (cx, cy) = win.cursor();
        let aim_world = game
            .camera()
            .to_world(cx / upscale as f32, cy / upscale as f32);
        let input = Input {
            mv,
            aim: aim_world.sub(game.player_pos()),
            fire: win.key(VK_SPACE) || win.key(VK_LBUTTON),
            restart: win.key(b'R' as i32),
        };

        game.step(&input, dt);
        submit_frame(
            dev,
            &world,
            &pol,
            &game,
            w,
            h,
            Some(world.targets[frame_no % 2]),
        );
        if frame_no > 0 {
            let mut fb = read_target(dev, world.targets[(frame_no - 1) % 2]);
            game.render_hud(&mut fb, w as usize, h as usize);
            win.present(&fb, w as usize, h as usize);
        }
        frame_no += 1;

        fps_frames += 1;
        if now - fps_t >= 0.5 {
            let fps = fps_frames as f64 / (now - fps_t);
            fps_frames = 0;
            fps_t = now;
            win.set_title(&format!(
                "ATOM ARENA — atom-vk — {fps:.0} fps — score {} wave {} lives {}",
                game.score, game.wave, game.lives
            ));
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}
