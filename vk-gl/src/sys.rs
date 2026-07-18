//! The FFI leaf: WGL context + GL 4.3 entry points, loaded by hand.
//! Zero external crates — the same quarantine pattern as the game's
//! win32 presentation leaf. Everything unsafe in the backend lives here.
#![allow(unsafe_code)]

use std::ffi::OsStr;
use std::os::windows::ffi::OsStrExt;

pub type GLenum = u32;
pub type GLuint = u32;
pub type GLint = i32;
pub type GLsizei = i32;
pub type GLchar = i8;
pub type GLsizeiptr = isize;
pub type GLintptr = isize;
pub type GLbitfield = u32;
pub type GLboolean = u8;
pub type GLsync = isize;

pub const GL_COMPUTE_SHADER: GLenum = 0x91B9;
pub const GL_SHADER_STORAGE_BUFFER: GLenum = 0x90D2;
pub const GL_UNIFORM_BUFFER: GLenum = 0x8A11;
#[allow(dead_code)] // some constants are reference documentation of the driver quirk
pub const GL_COPY_READ_BUFFER: GLenum = 0x8A36;
#[allow(dead_code)]
pub const GL_COPY_WRITE_BUFFER: GLenum = 0x8A37;
pub const GL_STREAM_DRAW: GLenum = 0x88E0;
pub const GL_DYNAMIC_COPY: GLenum = 0x88EA;
pub const GL_MAP_READ_BIT: GLbitfield = 0x0001;
pub const GL_COMPILE_STATUS: GLenum = 0x8B81;
pub const GL_LINK_STATUS: GLenum = 0x8B82;
pub const GL_INFO_LOG_LENGTH: GLenum = 0x8B84;
pub const GL_SHADER_STORAGE_BARRIER_BIT: GLbitfield = 0x2000;
pub const GL_BUFFER_UPDATE_BARRIER_BIT: GLbitfield = 0x0200;
pub const GL_SYNC_GPU_COMMANDS_COMPLETE: GLenum = 0x9117;
pub const GL_ALREADY_SIGNALED: GLenum = 0x911A;
pub const GL_CONDITION_SATISFIED: GLenum = 0x911C;
pub const GL_RENDERER: GLenum = 0x1F01;

type Pfn = *const core::ffi::c_void;

pub type WglCreateContextAttribsARB = unsafe extern "system" fn(isize, isize, *const i32) -> isize;
pub type GlCreateShader = unsafe extern "system" fn(GLenum) -> GLuint;
pub type GlShaderSource =
    unsafe extern "system" fn(GLuint, GLsizei, *const *const GLchar, *const GLint);
pub type GlCompileShader = unsafe extern "system" fn(GLuint);
pub type GlGetShaderiv = unsafe extern "system" fn(GLuint, GLenum, *mut GLint);
pub type GlGetShaderInfoLog = unsafe extern "system" fn(GLuint, GLsizei, *mut GLsizei, *mut GLchar);
pub type GlCreateProgram = unsafe extern "system" fn() -> GLuint;
pub type GlAttachShader = unsafe extern "system" fn(GLuint, GLuint);
pub type GlLinkProgram = unsafe extern "system" fn(GLuint);
pub type GlGetProgramiv = unsafe extern "system" fn(GLuint, GLenum, *mut GLint);
pub type GlGetProgramInfoLog =
    unsafe extern "system" fn(GLuint, GLsizei, *mut GLsizei, *mut GLchar);
pub type GlUseProgram = unsafe extern "system" fn(GLuint);
pub type GlGenBuffers = unsafe extern "system" fn(GLsizei, *mut GLuint);
pub type GlBindBuffer = unsafe extern "system" fn(GLenum, GLuint);
pub type GlBufferData =
    unsafe extern "system" fn(GLenum, GLsizeiptr, *const core::ffi::c_void, GLenum);
pub type GlBufferSubData =
    unsafe extern "system" fn(GLenum, GLintptr, GLsizeiptr, *const core::ffi::c_void);
pub type GlBindBufferBase = unsafe extern "system" fn(GLenum, GLuint, GLuint);
pub type GlMapBufferRange =
    unsafe extern "system" fn(GLenum, GLintptr, GLsizeiptr, GLbitfield) -> *mut core::ffi::c_void;
pub type GlUnmapBuffer = unsafe extern "system" fn(GLenum) -> GLboolean;
pub type GlCopyBufferSubData =
    unsafe extern "system" fn(GLenum, GLenum, GLintptr, GLintptr, GLsizeiptr);
pub type GlDispatchCompute = unsafe extern "system" fn(GLuint, GLuint, GLuint);
pub type GlMemoryBarrier = unsafe extern "system" fn(GLbitfield);
pub type GlFenceSync = unsafe extern "system" fn(GLenum, GLbitfield) -> GLsync;
pub type GlClientWaitSync = unsafe extern "system" fn(GLsync, GLbitfield, u64) -> GLenum;
pub type GlDeleteSync = unsafe extern "system" fn(GLsync);
pub type GlGetError = unsafe extern "system" fn() -> GLenum;

#[link(name = "opengl32")]
extern "system" {
    fn wglCreateContext(dc: isize) -> isize;
    fn wglMakeCurrent(dc: isize, ctx: isize) -> i32;
    fn wglDeleteContext(ctx: isize) -> i32;
    fn wglGetProcAddress(name: *const u8) -> Pfn;
    fn glGetString(name: GLenum) -> *const u8;
    fn glGetError() -> GLenum;
}

#[link(name = "user32")]
extern "system" {
    fn GetModuleHandleW(name: *const u16) -> isize;
    fn RegisterClassExW(cls: *const WndClassEx) -> u16;
    fn CreateWindowExW(
        ex: u32,
        class: *const u16,
        title: *const u16,
        style: u32,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        parent: isize,
        menu: isize,
        inst: isize,
        param: *mut core::ffi::c_void,
    ) -> isize;
    fn DefWindowProcW(hwnd: isize, msg: u32, wp: usize, lp: isize) -> isize;
    fn GetDC(hwnd: isize) -> isize;
    fn ReleaseDC(hwnd: isize, dc: isize) -> i32;
    fn DestroyWindow(hwnd: isize) -> i32;
    fn LoadCursorW(inst: isize, name: usize) -> isize;
}

#[link(name = "gdi32")]
extern "system" {
    fn ChoosePixelFormat(dc: isize, pfd: *const PixelFormatDescriptor) -> i32;
    fn SetPixelFormat(dc: isize, fmt: i32, pfd: *const PixelFormatDescriptor) -> i32;
}

#[repr(C)]
struct WndClassEx {
    size: u32,
    style: u32,
    wnd_proc: usize,
    cls_extra: i32,
    wnd_extra: i32,
    inst: isize,
    icon: isize,
    cursor: isize,
    brush: isize,
    menu: *const u16,
    class: *const u16,
    icon_sm: isize,
}

#[repr(C)]
struct PixelFormatDescriptor {
    size: u16,
    version: u16,
    flags: u32,
    pixel_type: u8,
    color_bits: u8,
    red_bits: u8,
    red_shift: u8,
    green_bits: u8,
    green_shift: u8,
    blue_bits: u8,
    blue_shift: u8,
    alpha_bits: u8,
    alpha_shift: u8,
    accum_bits: u8,
    accum_red: u8,
    accum_green: u8,
    accum_blue: u8,
    accum_alpha: u8,
    depth_bits: u8,
    stencil_bits: u8,
    aux_buffers: u8,
    layer_type: u8,
    reserved: u8,
    layer_mask: u32,
    visible_mask: u32,
    damage_mask: u32,
}

pub struct GlFns {
    pub get_error: GlGetError,
    pub create_shader: GlCreateShader,
    pub shader_source: GlShaderSource,
    pub compile_shader: GlCompileShader,
    pub get_shaderiv: GlGetShaderiv,
    pub shader_info_log: GlGetShaderInfoLog,
    pub create_program: GlCreateProgram,
    pub attach_shader: GlAttachShader,
    pub link_program: GlLinkProgram,
    pub get_programiv: GlGetProgramiv,
    pub program_info_log: GlGetProgramInfoLog,
    pub use_program: GlUseProgram,
    pub gen_buffers: GlGenBuffers,
    pub bind_buffer: GlBindBuffer,
    pub buffer_data: GlBufferData,
    pub buffer_sub_data: GlBufferSubData,
    pub bind_buffer_base: GlBindBufferBase,
    pub map_buffer_range: GlMapBufferRange,
    pub unmap_buffer: GlUnmapBuffer,
    pub copy_buffer_sub_data: GlCopyBufferSubData,
    pub dispatch_compute: GlDispatchCompute,
    pub memory_barrier: GlMemoryBarrier,
    pub fence_sync: GlFenceSync,
    pub client_wait_sync: GlClientWaitSync,
    pub delete_sync: GlDeleteSync,
}

/// A current GL 4.3 core context on a hidden window, plus every entry
/// point the backend calls. Single-thread affinity: WGL contexts are
/// current on the thread that created them.
pub struct Context {
    hwnd: isize,
    dc: isize,
    glctx: isize,
    pub fns: GlFns,
    pub renderer: String,
}

fn wide(s: &str) -> Vec<u16> {
    OsStr::new(s)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect()
}

fn cstr(s: &str) -> Vec<u8> {
    s.bytes().chain(std::iter::once(0)).collect()
}

unsafe fn load<T>(name: &str) -> Result<T, String> {
    let p = wglGetProcAddress(cstr(name).as_ptr());
    if p.is_null() {
        return Err(format!("wglGetProcAddress({name}) failed"));
    }
    Ok(std::mem::transmute_copy::<Pfn, T>(&p))
}

extern "system" fn wnd_proc(hwnd: isize, msg: u32, wp: usize, lp: isize) -> isize {
    unsafe { DefWindowProcW(hwnd, msg, wp, lp) }
}

impl Context {
    pub fn new() -> Result<Context, String> {
        unsafe {
            let inst = GetModuleHandleW(std::ptr::null());
            let class = wide("AtomVkGlClass");
            let wc = WndClassEx {
                size: std::mem::size_of::<WndClassEx>() as u32,
                style: 0,
                wnd_proc: wnd_proc as *const () as usize,
                cls_extra: 0,
                wnd_extra: 0,
                inst,
                icon: 0,
                cursor: LoadCursorW(0, 32512),
                brush: 0,
                menu: std::ptr::null(),
                class: class.as_ptr(),
                icon_sm: 0,
            };
            if RegisterClassExW(&wc) == 0 {
                return Err("RegisterClassExW failed".into());
            }
            const WS_POPUP: u32 = 0x8000_0000;
            let hwnd = CreateWindowExW(
                0,
                class.as_ptr(),
                wide("atom-vk-gl").as_ptr(),
                WS_POPUP,
                0,
                0,
                1,
                1,
                0,
                0,
                inst,
                std::ptr::null_mut(),
            );
            if hwnd == 0 {
                return Err("CreateWindowExW failed".into());
            }
            let dc = GetDC(hwnd);
            let pfd = PixelFormatDescriptor {
                size: 40,
                version: 1,
                flags: 0x0000_0004 | 0x0000_0020, // DRAW_TO_WINDOW | SUPPORT_OPENGL
                pixel_type: 0,
                color_bits: 32,
                red_bits: 0,
                red_shift: 0,
                green_bits: 0,
                green_shift: 0,
                blue_bits: 0,
                blue_shift: 0,
                alpha_bits: 0,
                alpha_shift: 0,
                accum_bits: 0,
                accum_red: 0,
                accum_green: 0,
                accum_blue: 0,
                accum_alpha: 0,
                depth_bits: 24,
                stencil_bits: 8,
                aux_buffers: 0,
                layer_type: 0,
                reserved: 0,
                layer_mask: 0,
                visible_mask: 0,
                damage_mask: 0,
            };
            let fmt = ChoosePixelFormat(dc, &pfd);
            if fmt == 0 || SetPixelFormat(dc, fmt, &pfd) == 0 {
                return Err("SetPixelFormat failed".into());
            }
            let legacy = wglCreateContext(dc);
            if legacy == 0 || wglMakeCurrent(dc, legacy) == 0 {
                return Err("legacy wgl context failed".into());
            }
            let attribs: WglCreateContextAttribsARB = load("wglCreateContextAttribsARB")?;
            let attrs = [
                0x2091, 4, // MAJOR = 4
                0x2092, 3, // MINOR = 3
                0x9126, 1, // PROFILE_MASK = CORE
                0, 0,
            ];
            let glctx = attribs(dc, 0, attrs.as_ptr());
            if glctx == 0 || wglMakeCurrent(dc, glctx) == 0 {
                return Err("GL 4.3 core context unavailable".into());
            }
            wglDeleteContext(legacy);

            let fns = GlFns {
                get_error: glGetError,
                create_shader: load("glCreateShader")?,
                shader_source: load("glShaderSource")?,
                compile_shader: load("glCompileShader")?,
                get_shaderiv: load("glGetShaderiv")?,
                shader_info_log: load("glGetShaderInfoLog")?,
                create_program: load("glCreateProgram")?,
                attach_shader: load("glAttachShader")?,
                link_program: load("glLinkProgram")?,
                get_programiv: load("glGetProgramiv")?,
                program_info_log: load("glGetProgramInfoLog")?,
                use_program: load("glUseProgram")?,
                gen_buffers: load("glGenBuffers")?,
                bind_buffer: load("glBindBuffer")?,
                buffer_data: load("glBufferData")?,
                buffer_sub_data: load("glBufferSubData")?,
                bind_buffer_base: load("glBindBufferBase")?,
                map_buffer_range: load("glMapBufferRange")?,
                unmap_buffer: load("glUnmapBuffer")?,
                copy_buffer_sub_data: load("glCopyBufferSubData")?,
                dispatch_compute: load("glDispatchCompute")?,
                memory_barrier: load("glMemoryBarrier")?,
                fence_sync: load("glFenceSync")?,
                client_wait_sync: load("glClientWaitSync")?,
                delete_sync: load("glDeleteSync")?,
            };
            let renderer = {
                let p = glGetString(GL_RENDERER);
                if p.is_null() {
                    "unknown GL renderer".to_string()
                } else {
                    std::ffi::CStr::from_ptr(p as *const i8)
                        .to_string_lossy()
                        .into_owned()
                }
            };
            Ok(Context {
                hwnd,
                dc,
                glctx,
                fns,
                renderer,
            })
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            wglMakeCurrent(0, 0);
            wglDeleteContext(self.glctx);
            ReleaseDC(self.hwnd, self.dc);
            DestroyWindow(self.hwnd);
        }
    }
}
