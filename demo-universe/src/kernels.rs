//! The universum substrate's laws as one moment-step kernel per backend
//! language. Mirrors field.rs @ 744 lines (2026-07-19): boundary
//! stimulus, phase-aligned pairwise exchange, spectral radiation,
//! dissipation, Hebbian permeability formation with erosion, phase
//! advance, and adaptive coupling (compound/imprint/disuse). Two
//! ping-ponged states make each moment a pure Jacobi step — no
//! synchronization inside a moment.
//!
//! Site layout (3 vec4s): s0 = (energy.rgb, permeability),
//! s1 = (trace.rgb, activity), s2 = (coupling.rgb, phase).
//!
//! GL bindings: 0=U, 1..3 = in s0..s2, 4..6 = out s0..s2.
//! WGSL bindings: 0=U, 2..4 = in, 5..7 = out.

const LAWS_GLSL: &str = r"
layout(std140, binding = 0) uniform U {
    vec4 dims;   // x=width, y=height, z=mode (0 adaptive, 1 fixed, 2 inert), w=age
    vec4 rates1; // diffusion, gradient, formation, erosion
    vec4 rates2; // coupling_formation, coupling_erosion, radiation, dissipation
} u;

layout(std430, binding = 1) readonly buffer In0 { vec4 in0[]; };
layout(std430, binding = 2) readonly buffer In1 { vec4 in1[]; };
layout(std430, binding = 3) readonly buffer In2 { vec4 in2[]; };
layout(std430, binding = 4) writeonly buffer Out0 { vec4 out0[]; };
layout(std430, binding = 5) writeonly buffer Out1 { vec4 out1[]; };
layout(std430, binding = 6) writeonly buffer Out2 { vec4 out2[]; };

const float TAU = 6.283185307179586;
const float PI = 3.141592653589793;

vec3 stimulus(float y) {
    float h = u.dims.y;
    float pulse = u.dims.w * 0.037;
    vec3 s = vec3(0.0);
    float c0 = round((0.23 + sin(pulse) * 0.025) * h);
    float c1 = round((0.50 + sin(pulse * 0.73 + 1.2) * 0.035) * h);
    float c2 = round((0.77 + sin(pulse * 0.51 + 2.4) * 0.025) * h);
    float d0 = abs(y - c0);
    if (d0 <= 2.0) { s += vec3(0.92, 0.08, 0.03) * (0.16 * (3.0 - d0) / 3.0); }
    float d1 = abs(y - c1);
    if (d1 <= 2.0) { s += vec3(0.05, 0.88, 0.12) * (0.16 * (3.0 - d1) / 3.0); }
    float d2 = abs(y - c2);
    if (d2 <= 2.0) { s += vec3(0.04, 0.13, 0.96) * (0.16 * (3.0 - d2) / 3.0); }
    return s;
}

vec3 energy_at(int idx, int x, float y) {
    vec3 e = in0[idx].xyz;
    if (x == 1) { e += stimulus(y); }
    return e;
}

vec3 pair_flow(vec3 ea, float pa, float pha, vec3 eb, float pb, float phb, bool xdir) {
    float edge_phase = xdir ? 0.0 : PI * 0.5;
    float alignment = 0.5 + 0.5 * cos(pha - phb + edge_phase);
    float material = sqrt(pa * pb);
    float conductance = (0.14 + 0.86 * material) * (0.42 + 0.58 * alignment);
    vec3 relaxation = (ea - eb) * (u.rates1.x * conductance);
    vec3 drift = xdir ? ea * (u.rates1.y * conductance) : vec3(0.0);
    return relaxation + drift;
}

float mag3(vec3 v) { return abs(v.x) + abs(v.y) + abs(v.z); }

float spectral_turn(vec3 s) {
    float total = s.x + s.y + s.z;
    if (total <= 0.000001) { return 0.0; }
    float r = s.x / total;
    float g = s.y / total;
    float b = s.z / total;
    return (r - b) * 0.72 + (g - 0.5 * (r + b)) * 0.38;
}

void main() {
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    int W = int(u.dims.x);
    int H = int(u.dims.y);
    if (gid.x >= W || gid.y >= H) { return; }
    int x = gid.x;
    float y = float(gid.y);
    int i = gid.y * W + gid.x;

    vec3 self_e = energy_at(i, x, y);
    float self_p = in0[i].w;
    float self_ph = in2[i].w;

    vec3 change = vec3(0.0);
    float lf = 0.0;
    // accumulation order mirrors the CPU sweep: up, left, right, down
    if (gid.y > 0) {
        int n = i - W;
        vec3 f = pair_flow(energy_at(n, x, y - 1.0), in0[n].w, in2[n].w, self_e, self_p, self_ph, false);
        change += f;
        lf += mag3(f);
    }
    if (x > 0) {
        int n = i - 1;
        vec3 f = pair_flow(energy_at(n, x - 1, y), in0[n].w, in2[n].w, self_e, self_p, self_ph, true);
        change += f;
        lf += mag3(f);
    }
    if (x + 1 < W) {
        int n = i + 1;
        vec3 f = pair_flow(self_e, self_p, self_ph, energy_at(n, x + 1, y), in0[n].w, in2[n].w, true);
        change -= f;
        lf += mag3(f);
    }
    if (gid.y + 1 < H) {
        int n = i + W;
        vec3 f = pair_flow(self_e, self_p, self_ph, energy_at(n, x, y + 1.0), in0[n].w, in2[n].w, false);
        change -= f;
        lf += mag3(f);
    }

    // flows see the STIMULATED energies (CPU stimulates before
    // exchange); the application clamps raw+stimulus+change TOGETHER,
    // exactly like the CPU's (site.energy + change).clamp_nonnegative()
    // after stimulate_boundary mutated the sites in place.
    vec3 applied = in0[i].xyz + change;
    if (x == 1) { applied += stimulus(y); }
    vec3 energy = max(applied, vec3(0.0));

    vec3 coupling = in2[i].xyz;
    vec3 emitted = energy * coupling * u.rates2.z;
    energy -= emitted;
    vec3 trace = in1[i].xyz + emitted;

    vec3 lost = energy * u.rates2.w;
    energy -= lost;

    float activity = in1[i].w * 0.94 + min(lf * 2.5, 1.0) * 0.06;
    float growth = u.rates1.z * activity * (1.0 - self_p);
    float decay = u.rates1.w * max(self_p - 0.05, 0.0);
    float perm = clamp(self_p + growth - decay, 0.025, 1.0);
    float phase = mod(self_ph + activity * spectral_turn(energy) * 0.024, TAU);

    if (u.dims.z < 0.5) {
        float nx = float(x) / float(W - 1);
        float access = clamp((nx - 0.40) / 0.20, 0.0, 1.0);
        if (access <= 0.0) {
            coupling = vec3(0.0);
        } else {
            vec3 signal = energy + emitted;
            float st = signal.x + signal.y + signal.z;
            vec3 shape = st > 0.000001 ? signal / st : vec3(0.0);
            float conversion_drive = min(mag3(emitted) * 28.0, 1.0);
            float signal_drive = min(mag3(signal) * 0.24, 1.0);
            float transport_drive = min(lf * 0.42, 1.0);
            float drive = clamp(signal_drive * (activity * 0.70 + transport_drive * 0.25) + conversion_drive * 0.05, 0.0, 1.0);
            float target_amplitude = access * clamp(0.18 + perm * 0.82, 0.0, 1.0);
            vec3 target = shape * target_amplitude;
            float peak = max(coupling.x, max(coupling.y, coupling.z));
            float readiness = clamp(peak * 260.0, 0.0, 1.0);
            float compound_drive = drive * (0.10 + readiness * 0.90);
            float compound = 1.0 + u.rates2.x * compound_drive * (1.0 + peak * 28.0);
            vec3 imprint = target * (u.rates2.x * drive * 0.16);
            coupling = coupling * min(compound, 1.32) + imprint;
            coupling = min(coupling, max(target, vec3(0.001)));
            float disuse = u.rates2.y * clamp(1.0 - activity * 0.72, 0.0, 1.0);
            coupling = clamp(coupling * (1.0 - disuse), 0.0, 1.0);
        }
    }

    out0[i] = vec4(energy, perm);
    out1[i] = vec4(trace, activity);
    out2[i] = vec4(coupling, phase);
}
";

const LAWS_WGSL: &str = r"
struct U {
    dims: vec4<f32>,
    rates1: vec4<f32>,
    rates2: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(2) var<storage, read> in0: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> in1: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> in2: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> out0: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> out1: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> out2: array<vec4<f32>>;

const TAU = 6.283185307179586;
const PI = 3.141592653589793;

fn stimulus(y: f32) -> vec3<f32> {
    let h = u.dims.y;
    let pulse = u.dims.w * 0.037;
    var s = vec3<f32>(0.0, 0.0, 0.0);
    let c0 = round((0.23 + sin(pulse) * 0.025) * h);
    let c1 = round((0.50 + sin(pulse * 0.73 + 1.2) * 0.035) * h);
    let c2 = round((0.77 + sin(pulse * 0.51 + 2.4) * 0.025) * h);
    let d0 = abs(y - c0);
    if d0 <= 2.0 { s += vec3<f32>(0.92, 0.08, 0.03) * (0.16 * (3.0 - d0) / 3.0); }
    let d1 = abs(y - c1);
    if d1 <= 2.0 { s += vec3<f32>(0.05, 0.88, 0.12) * (0.16 * (3.0 - d1) / 3.0); }
    let d2 = abs(y - c2);
    if d2 <= 2.0 { s += vec3<f32>(0.04, 0.13, 0.96) * (0.16 * (3.0 - d2) / 3.0); }
    return s;
}

fn energy_at(idx: u32, x: u32, y: f32) -> vec3<f32> {
    var e = in0[idx].xyz;
    if x == 1u { e += stimulus(y); }
    return e;
}

fn pair_flow(ea: vec3<f32>, pa: f32, pha: f32, eb: vec3<f32>, pb: f32, phb: f32, xdir: bool) -> vec3<f32> {
    let edge_phase = select(PI * 0.5, 0.0, xdir);
    let alignment = 0.5 + 0.5 * cos(pha - phb + edge_phase);
    let material = sqrt(pa * pb);
    let conductance = (0.14 + 0.86 * material) * (0.42 + 0.58 * alignment);
    let relaxation = (ea - eb) * (u.rates1.x * conductance);
    var drift = vec3<f32>(0.0, 0.0, 0.0);
    if xdir { drift = ea * (u.rates1.y * conductance); }
    return relaxation + drift;
}

fn mag3(v: vec3<f32>) -> f32 { return abs(v.x) + abs(v.y) + abs(v.z); }

fn spectral_turn(s: vec3<f32>) -> f32 {
    let total = s.x + s.y + s.z;
    if total <= 0.000001 { return 0.0; }
    let r = s.x / total;
    let g = s.y / total;
    let b = s.z / total;
    return (r - b) * 0.72 + (g - 0.5 * (r + b)) * 0.38;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let W = u32(u.dims.x);
    let H = u32(u.dims.y);
    if gid3.x >= W || gid3.y >= H { return; }
    let x = gid3.x;
    let y = f32(gid3.y);
    let i = gid3.y * W + gid3.x;

    let self_e = energy_at(i, x, y);
    let self_p = in0[i].w;
    let self_ph = in2[i].w;

    var change = vec3<f32>(0.0, 0.0, 0.0);
    var lf = 0.0;
    if gid3.y > 0u {
        let n = i - W;
        let f = pair_flow(energy_at(n, x, y - 1.0), in0[n].w, in2[n].w, self_e, self_p, self_ph, false);
        change += f;
        lf += mag3(f);
    }
    if x > 0u {
        let n = i - 1u;
        let f = pair_flow(energy_at(n, x - 1u, y), in0[n].w, in2[n].w, self_e, self_p, self_ph, true);
        change += f;
        lf += mag3(f);
    }
    if x + 1u < W {
        let n = i + 1u;
        let f = pair_flow(self_e, self_p, self_ph, energy_at(n, x + 1u, y), in0[n].w, in2[n].w, true);
        change -= f;
        lf += mag3(f);
    }
    if gid3.y + 1u < H {
        let n = i + W;
        let f = pair_flow(self_e, self_p, self_ph, energy_at(n, x, y + 1.0), in0[n].w, in2[n].w, false);
        change -= f;
        lf += mag3(f);
    }

    var applied = in0[i].xyz + change;
    if x == 1u { applied += stimulus(y); }
    var energy = max(applied, vec3<f32>(0.0, 0.0, 0.0));

    var coupling = in2[i].xyz;
    let emitted = energy * coupling * u.rates2.z;
    energy -= emitted;
    let trace = in1[i].xyz + emitted;

    let lost = energy * u.rates2.w;
    energy -= lost;

    let activity = in1[i].w * 0.94 + min(lf * 2.5, 1.0) * 0.06;
    let growth = u.rates1.z * activity * (1.0 - self_p);
    let decay = u.rates1.w * max(self_p - 0.05, 0.0);
    let perm = clamp(self_p + growth - decay, 0.025, 1.0);
    let phase = fract((self_ph + activity * spectral_turn(energy) * 0.024) / TAU) * TAU;

    if u.dims.z < 0.5 {
        let nx = f32(x) / f32(W - 1u);
        let access = clamp((nx - 0.40) / 0.20, 0.0, 1.0);
        if access <= 0.0 {
            coupling = vec3<f32>(0.0, 0.0, 0.0);
        } else {
            let signal = energy + emitted;
            let st = signal.x + signal.y + signal.z;
            var shape = vec3<f32>(0.0, 0.0, 0.0);
            if st > 0.000001 { shape = signal / st; }
            let conversion_drive = min(mag3(emitted) * 28.0, 1.0);
            let signal_drive = min(mag3(signal) * 0.24, 1.0);
            let transport_drive = min(lf * 0.42, 1.0);
            let drive = clamp(signal_drive * (activity * 0.70 + transport_drive * 0.25) + conversion_drive * 0.05, 0.0, 1.0);
            let target_amplitude = access * clamp(0.18 + perm * 0.82, 0.0, 1.0);
            let tgt = shape * target_amplitude;
            let peak = max(coupling.x, max(coupling.y, coupling.z));
            let readiness = clamp(peak * 260.0, 0.0, 1.0);
            let compound_drive = drive * (0.10 + readiness * 0.90);
            let compound = 1.0 + u.rates2.x * compound_drive * (1.0 + peak * 28.0);
            let imprint = tgt * (u.rates2.x * drive * 0.16);
            coupling = coupling * min(compound, 1.32) + imprint;
            coupling = min(coupling, max(tgt, vec3<f32>(0.001, 0.001, 0.001)));
            let disuse = u.rates2.y * clamp(1.0 - activity * 0.72, 0.0, 1.0);
            coupling = clamp(coupling * (1.0 - disuse), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0));
        }
    }

    out0[i] = vec4<f32>(energy, perm);
    out1[i] = vec4<f32>(trace, activity);
    out2[i] = vec4<f32>(coupling, phase);
}
";

pub fn step_glsl() -> String {
    format!("#version 430\nlayout(local_size_x = 8, local_size_y = 8) in;\n{LAWS_GLSL}")
}

pub fn step_wgsl() -> String {
    LAWS_WGSL.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_generators_carry_the_laws() {
        for src in [step_glsl(), step_wgsl()] {
            assert!(src.contains("pair_flow"));
            assert!(src.contains("spectral_turn"));
            assert!(src.contains("compound"));
        }
        assert!(step_glsl().contains("#version 430"));
        assert!(step_wgsl().contains("@compute"));
    }
}
