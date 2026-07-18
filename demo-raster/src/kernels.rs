//! The graphics stage, as compute. One fused kernel per backend
//! language: transform → edge-test → nearest-z → shade, all per pixel.
//! Same algorithm, same layouts — the generator swap is text, not
//! architecture.
//!
//! History, kept honest: the first generator was one-thread-per-
//! triangle + atomicMin z-buffer (transform → attention bbox → atomic z
//! → resolve, 3 kernels + a clear pass). It was pixel-exact but SLOWER
//! THAN CPU at 1080p (38 fps vs 86) — 12 serial GPU threads each
//! looping ~2M px. The per-pixel generator below needs no z-buffer, no
//! atomics, no clear pass. Tile binning (the attention step) returns as
//! the generator when meshes get big.

fn glsl_preamble() -> &'static str {
    r#"
layout(std140, binding = 0) uniform U {
    mat4 mvp;
    vec4 params; // x: n_tris, y: width, z: height, w: tex size
} u;

layout(std430, binding = 1) readonly buffer Tris { vec4 tris[]; }; // 4 per tri: v0,v1,v2,color
layout(std430, binding = 2) readonly buffer Uvs { vec4 uvs[]; };   // 2 per tri: uv0.xy,uv1.xy | uv2.xy
layout(std430, binding = 3) readonly buffer Tex { uint tex[]; };   // packed u32 texels

float edgef(vec2 a, vec2 b, vec2 p) {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

vec3 unpack_rgb(uint v) {
    return vec3(float((v >> 16u) & 0xFFu), float((v >> 8u) & 0xFFu), float(v & 0xFFu)) / 255.0;
}

// the sampler: uv -> bilinear weighted texel fetch. wrap = repeat.
// combine() over the 4 taps with fractional weights.
vec3 tex_sample(vec2 uv) {
    uv = fract(uv);
    float size = u.params.w;
    vec2 t = uv * size - 0.5;
    vec2 base = floor(t);
    vec2 f = t - base;
    int sz = int(size);
    ivec2 b0 = (ivec2(base) + ivec2(sz)) % ivec2(sz);
    ivec2 b1 = (b0 + ivec2(1, 1)) % ivec2(sz);
    vec3 c00 = unpack_rgb(tex[uint(b0.y) * uint(size) + uint(b0.x)]);
    vec3 c10 = unpack_rgb(tex[uint(b0.y) * uint(size) + uint(b1.x)]);
    vec3 c01 = unpack_rgb(tex[uint(b1.y) * uint(size) + uint(b0.x)]);
    vec3 c11 = unpack_rgb(tex[uint(b1.y) * uint(size) + uint(b1.x)]);
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}
"#
}

fn wgsl_preamble() -> &'static str {
    r#"
struct U {
    mvp: mat4x4<f32>,
    params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(2) var<storage, read> tris: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> uvs: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> tex: array<u32>;

fn edgef(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> f32 {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

fn unpack_rgb(v: u32) -> vec3<f32> {
    return vec3<f32>(f32((v >> 16u) & 0xFFu), f32((v >> 8u) & 0xFFu), f32(v & 0xFFu)) / 255.0;
}

fn texel(c: vec2<i32>, sz: i32) -> vec3<f32> {
    let w = (c + vec2<i32>(sz, sz)) % vec2<i32>(sz, sz);
    return unpack_rgb(tex[u32(w.y) * u32(sz) + u32(w.x)]);
}

fn tex_sample(uv: vec2<f32>) -> vec3<f32> {
    let uvw = fract(uv);
    let size = u32(u.params.w);
    let t = uvw * f32(size) - 0.5;
    let base = floor(t);
    let f = t - base;
    let sz = i32(size);
    let b0 = vec2<i32>(base);
    let b1 = b0 + vec2<i32>(1, 1);
    let c00 = texel(vec2<i32>(b0.x, b0.y), sz);
    let c10 = texel(vec2<i32>(b1.x, b0.y), sz);
    let c01 = texel(vec2<i32>(b0.x, b1.y), sz);
    let c11 = texel(vec2<i32>(b1.x, b1.y), sz);
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}
"#
}

/// ONE thread per pixel loops all triangles, transforms them (redundant
/// but massively parallel), keeps the nearest, shades, writes. No
/// z-buffer, no atomics, no clear pass — every pixel has one owner.
pub fn fuse_glsl() -> String {
    let mut s = String::from("#version 430\nlayout(local_size_x = 8, local_size_y = 8) in;\n");
    s.push_str(glsl_preamble());
    s.push_str(
        r#"
layout(std430, binding = 7) writeonly buffer Pixels { uint pixels[]; };

void main() {
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    int W = int(u.params.y);
    int H = int(u.params.z);
    if (gid.x >= W || gid.y >= H) { return; }
    vec2 p = vec2(float(gid.x) + 0.5, float(gid.y) + 0.5);
    vec3 l = normalize(vec3(0.5, 0.8, -0.6));
    vec3 col = vec3(0.05, 0.06, 0.09);
    float best_z = 1e30;
    uint best_tri = 0xFFFFFFFFu;

    for (uint tri = 0u; tri < uint(u.params.x); tri++) {
        vec4 w0 = tris[tri * 4u + 0u];
        vec4 w1 = tris[tri * 4u + 1u];
        vec4 w2 = tris[tri * 4u + 2u];
        vec4 c0 = u.mvp * vec4(w0.xyz, 1.0);
        vec4 c1 = u.mvp * vec4(w1.xyz, 1.0);
        vec4 c2 = u.mvp * vec4(w2.xyz, 1.0);
        if (c0.w <= 0.0 || c1.w <= 0.0 || c2.w <= 0.0) { continue; }
        float Wf = u.params.y;
        float Hf = u.params.z;
        vec2 s0 = vec2((c0.x / c0.w * 0.5 + 0.5) * Wf, (0.5 - c0.y / c0.w * 0.5) * Hf);
        vec2 s1 = vec2((c1.x / c1.w * 0.5 + 0.5) * Wf, (0.5 - c1.y / c1.w * 0.5) * Hf);
        vec2 s2 = vec2((c2.x / c2.w * 0.5 + 0.5) * Wf, (0.5 - c2.y / c2.w * 0.5) * Hf);
        float z0 = c0.z / c0.w * 0.5 + 0.5;
        float z1 = c1.z / c1.w * 0.5 + 0.5;
        float z2 = c2.z / c2.w * 0.5 + 0.5;

        float e0 = edgef(s1, s2, p);
        float e1 = edgef(s2, s0, p);
        float e2 = edgef(s0, s1, p);
        bool pos = (e0 >= 0.0) && (e1 >= 0.0) && (e2 >= 0.0);
        bool neg = (e0 <= 0.0) && (e1 <= 0.0) && (e2 <= 0.0);
        if (!pos && !neg) { continue; }
        float area = abs(e0) + abs(e1) + abs(e2);
        if (area < 1e-9) { continue; }
        float z = (abs(e0) * z0 + abs(e1) * z1 + abs(e2) * z2) / area;
        if (z < best_z) {
            best_z = z;
            best_tri = tri;
        }
    }

    if (best_tri != 0xFFFFFFFFu) {
        vec4 w0 = tris[best_tri * 4u + 0u];
        vec4 w1 = tris[best_tri * 4u + 1u];
        vec4 w2 = tris[best_tri * 4u + 2u];
        vec3 n = normalize(cross(w1.xyz - w0.xyz, w2.xyz - w0.xyz));
        float shade = 0.25 + 0.75 * max(dot(n, l), 0.0);

        // barycentric uv at this pixel on the winning triangle
        vec4 c0 = u.mvp * vec4(w0.xyz, 1.0);
        vec4 c1 = u.mvp * vec4(w1.xyz, 1.0);
        vec4 c2 = u.mvp * vec4(w2.xyz, 1.0);
        float Wf = u.params.y;
        float Hf = u.params.z;
        vec2 s0 = vec2((c0.x / c0.w * 0.5 + 0.5) * Wf, (0.5 - c0.y / c0.w * 0.5) * Hf);
        vec2 s1 = vec2((c1.x / c1.w * 0.5 + 0.5) * Wf, (0.5 - c1.y / c1.w * 0.5) * Hf);
        vec2 s2 = vec2((c2.x / c2.w * 0.5 + 0.5) * Wf, (0.5 - c2.y / c2.w * 0.5) * Hf);
        float e0 = abs(edgef(s1, s2, p));
        float e1 = abs(edgef(s2, s0, p));
        float e2 = abs(edgef(s0, s1, p));
        float area = e0 + e1 + e2;
        vec2 uv = (e0 * uvs[best_tri * 2u].xy + e1 * uvs[best_tri * 2u].zw + e2 * uvs[best_tri * 2u + 1u].xy) / area;

        col = tris[best_tri * 4u + 3u].rgb * tex_sample(uv) * shade;
    }
    uint r = uint(clamp(col.r, 0.0, 1.0) * 255.0 + 0.5);
    uint g = uint(clamp(col.g, 0.0, 1.0) * 255.0 + 0.5);
    uint b = uint(clamp(col.b, 0.0, 1.0) * 255.0 + 0.5);
    pixels[uint(gid.y) * uint(W) + uint(gid.x)] = (r << 16u) | (g << 8u) | b;
}
"#,
    );
    s
}

pub fn fuse_wgsl() -> String {
    let mut s = wgsl_preamble().to_string();
    s.push_str(
        r#"
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let W = i32(u.params.y);
    let H = i32(u.params.z);
    if i32(gid3.x) >= W || i32(gid3.y) >= H { return; }
    let p = vec2<f32>(f32(gid3.x) + 0.5, f32(gid3.y) + 0.5);
    let l = normalize(vec3<f32>(0.5, 0.8, -0.6));
    var col = vec3<f32>(0.05, 0.06, 0.09);
    var best_z = 1e30;
    var best_tri = 0xFFFFFFFFu;

    for (var tri = 0u; tri < u32(u.params.x); tri = tri + 1u) {
        let w0 = tris[tri * 4u + 0u];
        let w1 = tris[tri * 4u + 1u];
        let w2 = tris[tri * 4u + 2u];
        let c0 = u.mvp * vec4<f32>(w0.xyz, 1.0);
        let c1 = u.mvp * vec4<f32>(w1.xyz, 1.0);
        let c2 = u.mvp * vec4<f32>(w2.xyz, 1.0);
        if c0.w <= 0.0 || c1.w <= 0.0 || c2.w <= 0.0 { continue; }
        let Wf = u.params.y;
        let Hf = u.params.z;
        let s0 = vec2<f32>((c0.x / c0.w * 0.5 + 0.5) * Wf, (0.5 - c0.y / c0.w * 0.5) * Hf);
        let s1 = vec2<f32>((c1.x / c1.w * 0.5 + 0.5) * Wf, (0.5 - c1.y / c1.w * 0.5) * Hf);
        let s2 = vec2<f32>((c2.x / c2.w * 0.5 + 0.5) * Wf, (0.5 - c2.y / c2.w * 0.5) * Hf);
        let z0 = c0.z / c0.w * 0.5 + 0.5;
        let z1 = c1.z / c1.w * 0.5 + 0.5;
        let z2 = c2.z / c2.w * 0.5 + 0.5;

        let e0 = edgef(s1, s2, p);
        let e1 = edgef(s2, s0, p);
        let e2 = edgef(s0, s1, p);
        let pos = (e0 >= 0.0) && (e1 >= 0.0) && (e2 >= 0.0);
        let neg = (e0 <= 0.0) && (e1 <= 0.0) && (e2 <= 0.0);
        if !pos && !neg { continue; }
        let area = abs(e0) + abs(e1) + abs(e2);
        if area < 1e-9 { continue; }
        let z = (abs(e0) * z0 + abs(e1) * z1 + abs(e2) * z2) / area;
        if z < best_z {
            best_z = z;
            best_tri = tri;
        }
    }

    if best_tri != 0xFFFFFFFFu {
        let w0 = tris[best_tri * 4u + 0u];
        let w1 = tris[best_tri * 4u + 1u];
        let w2 = tris[best_tri * 4u + 2u];
        let n = normalize(cross(w1.xyz - w0.xyz, w2.xyz - w0.xyz));
        let shade = 0.25 + 0.75 * max(dot(n, l), 0.0);

        let c0 = u.mvp * vec4<f32>(w0.xyz, 1.0);
        let c1 = u.mvp * vec4<f32>(w1.xyz, 1.0);
        let c2 = u.mvp * vec4<f32>(w2.xyz, 1.0);
        let Wf = u.params.y;
        let Hf = u.params.z;
        let s0 = vec2<f32>((c0.x / c0.w * 0.5 + 0.5) * Wf, (0.5 - c0.y / c0.w * 0.5) * Hf);
        let s1 = vec2<f32>((c1.x / c1.w * 0.5 + 0.5) * Wf, (0.5 - c1.y / c1.w * 0.5) * Hf);
        let s2 = vec2<f32>((c2.x / c2.w * 0.5 + 0.5) * Wf, (0.5 - c2.y / c2.w * 0.5) * Hf);
        let e0 = abs(edgef(s1, s2, p));
        let e1 = abs(edgef(s2, s0, p));
        let e2 = abs(edgef(s0, s1, p));
        let area = e0 + e1 + e2;
        let uv = (e0 * uvs[best_tri * 2u].xy + e1 * uvs[best_tri * 2u].zw + e2 * uvs[best_tri * 2u + 1u].xy) / area;

        col = tris[best_tri * 4u + 3u].rgb * tex_sample(uv) * shade;
    }
    let c = clamp(col, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(out_tex, vec2<i32>(gid3.xy), vec4<f32>(c.b, c.g, c.r, 1.0));
}
"#,
    );
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_generators_emit_the_fused_algorithm() {
        assert!(fuse_glsl().contains("edgef") && fuse_glsl().contains("best_z"));
        assert!(fuse_wgsl().contains("edgef") && fuse_wgsl().contains("best_z"));
        assert!(fuse_glsl().contains("#version 430"));
        assert!(fuse_wgsl().contains("@compute"));
    }
}
