// Generated by `wit-bindgen` 0.41.0. DO NOT EDIT!
// Options used:
//   * runtime_path: "wit_bindgen_rt"
#[rustfmt::skip]
#[allow(dead_code, clippy::all)]
pub mod exports {
    pub mod golem {
        pub mod it {
            #[allow(dead_code, async_fn_in_trait, unused_imports, clippy::all)]
            pub mod api {
                #[used]
                #[doc(hidden)]
                static __FORCE_SECTION_REF: fn() = super::super::super::super::__link_custom_section_describing_imports;
                use super::super::super::super::_rt;
                wit_bindgen_rt::bitflags::bitflags! {
                    #[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone, Copy)]
                    pub struct Permissions : u8 { const READ = 1 << 0; const WRITE = 1 <<
                    1; const EXEC = 1 << 2; const CLOSE = 1 << 3; }
                }
                #[derive(Clone)]
                pub struct Task {
                    pub name: _rt::String,
                    pub permissions: Permissions,
                }
                impl ::core::fmt::Debug for Task {
                    fn fmt(
                        &self,
                        f: &mut ::core::fmt::Formatter<'_>,
                    ) -> ::core::fmt::Result {
                        f.debug_struct("Task")
                            .field("name", &self.name)
                            .field("permissions", &self.permissions)
                            .finish()
                    }
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn _export_create_task_cabi<T: Guest>(
                    arg0: *mut u8,
                    arg1: usize,
                    arg2: i32,
                ) -> *mut u8 {
                    #[cfg(target_arch = "wasm32")] _rt::run_ctors_once();
                    let len0 = arg1;
                    let bytes0 = _rt::Vec::from_raw_parts(arg0.cast(), len0, len0);
                    let result1 = T::create_task(Task {
                        name: _rt::string_lift(bytes0),
                        permissions: Permissions::empty()
                            | Permissions::from_bits_retain(((arg2 as u8) << 0) as _),
                    });
                    let ptr2 = (&raw mut _RET_AREA.0).cast::<u8>();
                    let Task { name: name3, permissions: permissions3 } = result1;
                    let vec4 = (name3.into_bytes()).into_boxed_slice();
                    let ptr4 = vec4.as_ptr().cast::<u8>();
                    let len4 = vec4.len();
                    ::core::mem::forget(vec4);
                    *ptr2.add(::core::mem::size_of::<*const u8>()).cast::<usize>() = len4;
                    *ptr2.add(0).cast::<*mut u8>() = ptr4.cast_mut();
                    let flags5 = permissions3;
                    *ptr2.add(2 * ::core::mem::size_of::<*const u8>()).cast::<u8>() = ((flags5
                        .bits() >> 0) as i32) as u8;
                    ptr2
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn __post_return_create_task<T: Guest>(arg0: *mut u8) {
                    let l0 = *arg0.add(0).cast::<*mut u8>();
                    let l1 = *arg0
                        .add(::core::mem::size_of::<*const u8>())
                        .cast::<usize>();
                    _rt::cabi_dealloc(l0, l1, 1);
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn _export_get_tasks_cabi<T: Guest>() -> *mut u8 {
                    #[cfg(target_arch = "wasm32")] _rt::run_ctors_once();
                    let result0 = T::get_tasks();
                    let ptr1 = (&raw mut _RET_AREA.0).cast::<u8>();
                    let vec5 = result0;
                    let len5 = vec5.len();
                    let layout5 = _rt::alloc::Layout::from_size_align_unchecked(
                        vec5.len() * (3 * ::core::mem::size_of::<*const u8>()),
                        ::core::mem::size_of::<*const u8>(),
                    );
                    let result5 = if layout5.size() != 0 {
                        let ptr = _rt::alloc::alloc(layout5).cast::<u8>();
                        if ptr.is_null() {
                            _rt::alloc::handle_alloc_error(layout5);
                        }
                        ptr
                    } else {
                        ::core::ptr::null_mut()
                    };
                    for (i, e) in vec5.into_iter().enumerate() {
                        let base = result5
                            .add(i * (3 * ::core::mem::size_of::<*const u8>()));
                        {
                            let Task { name: name2, permissions: permissions2 } = e;
                            let vec3 = (name2.into_bytes()).into_boxed_slice();
                            let ptr3 = vec3.as_ptr().cast::<u8>();
                            let len3 = vec3.len();
                            ::core::mem::forget(vec3);
                            *base
                                .add(::core::mem::size_of::<*const u8>())
                                .cast::<usize>() = len3;
                            *base.add(0).cast::<*mut u8>() = ptr3.cast_mut();
                            let flags4 = permissions2;
                            *base
                                .add(2 * ::core::mem::size_of::<*const u8>())
                                .cast::<u8>() = ((flags4.bits() >> 0) as i32) as u8;
                        }
                    }
                    *ptr1.add(::core::mem::size_of::<*const u8>()).cast::<usize>() = len5;
                    *ptr1.add(0).cast::<*mut u8>() = result5;
                    ptr1
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn __post_return_get_tasks<T: Guest>(arg0: *mut u8) {
                    let l0 = *arg0.add(0).cast::<*mut u8>();
                    let l1 = *arg0
                        .add(::core::mem::size_of::<*const u8>())
                        .cast::<usize>();
                    let base4 = l0;
                    let len4 = l1;
                    for i in 0..len4 {
                        let base = base4
                            .add(i * (3 * ::core::mem::size_of::<*const u8>()));
                        {
                            let l2 = *base.add(0).cast::<*mut u8>();
                            let l3 = *base
                                .add(::core::mem::size_of::<*const u8>())
                                .cast::<usize>();
                            _rt::cabi_dealloc(l2, l3, 1);
                        }
                    }
                    _rt::cabi_dealloc(
                        base4,
                        len4 * (3 * ::core::mem::size_of::<*const u8>()),
                        ::core::mem::size_of::<*const u8>(),
                    );
                }
                pub trait Guest {
                    fn create_task(input: Task) -> Task;
                    fn get_tasks() -> _rt::Vec<Task>;
                }
                #[doc(hidden)]
                macro_rules! __export_golem_it_api_cabi {
                    ($ty:ident with_types_in $($path_to_types:tt)*) => {
                        const _ : () = { #[unsafe (export_name =
                        "golem:it/api#create-task")] unsafe extern "C" fn
                        export_create_task(arg0 : * mut u8, arg1 : usize, arg2 : i32,) ->
                        * mut u8 { unsafe { $($path_to_types)*::
                        _export_create_task_cabi::<$ty > (arg0, arg1, arg2) } } #[unsafe
                        (export_name = "cabi_post_golem:it/api#create-task")] unsafe
                        extern "C" fn _post_return_create_task(arg0 : * mut u8,) { unsafe
                        { $($path_to_types)*:: __post_return_create_task::<$ty > (arg0) }
                        } #[unsafe (export_name = "golem:it/api#get-tasks")] unsafe
                        extern "C" fn export_get_tasks() -> * mut u8 { unsafe {
                        $($path_to_types)*:: _export_get_tasks_cabi::<$ty > () } }
                        #[unsafe (export_name = "cabi_post_golem:it/api#get-tasks")]
                        unsafe extern "C" fn _post_return_get_tasks(arg0 : * mut u8,) {
                        unsafe { $($path_to_types)*:: __post_return_get_tasks::<$ty >
                        (arg0) } } };
                    };
                }
                #[doc(hidden)]
                pub(crate) use __export_golem_it_api_cabi;
                #[cfg_attr(target_pointer_width = "64", repr(align(8)))]
                #[cfg_attr(target_pointer_width = "32", repr(align(4)))]
                struct _RetArea(
                    [::core::mem::MaybeUninit<
                        u8,
                    >; 3 * ::core::mem::size_of::<*const u8>()],
                );
                static mut _RET_AREA: _RetArea = _RetArea(
                    [::core::mem::MaybeUninit::uninit(); 3
                        * ::core::mem::size_of::<*const u8>()],
                );
            }
        }
    }
}
#[rustfmt::skip]
mod _rt {
    #![allow(dead_code, clippy::all)]
    pub use alloc_crate::string::String;
    #[cfg(target_arch = "wasm32")]
    pub fn run_ctors_once() {
        wit_bindgen_rt::run_ctors_once();
    }
    pub use alloc_crate::vec::Vec;
    pub unsafe fn string_lift(bytes: Vec<u8>) -> String {
        if cfg!(debug_assertions) {
            String::from_utf8(bytes).unwrap()
        } else {
            String::from_utf8_unchecked(bytes)
        }
    }
    pub unsafe fn cabi_dealloc(ptr: *mut u8, size: usize, align: usize) {
        if size == 0 {
            return;
        }
        let layout = alloc::Layout::from_size_align_unchecked(size, align);
        alloc::dealloc(ptr, layout);
    }
    pub use alloc_crate::alloc;
    extern crate alloc as alloc_crate;
}
/// Generates `#[unsafe(no_mangle)]` functions to export the specified type as
/// the root implementation of all generated traits.
///
/// For more information see the documentation of `wit_bindgen::generate!`.
///
/// ```rust
/// # macro_rules! export{ ($($t:tt)*) => (); }
/// # trait Guest {}
/// struct MyType;
///
/// impl Guest for MyType {
///     // ...
/// }
///
/// export!(MyType);
/// ```
#[allow(unused_macros)]
#[doc(hidden)]
macro_rules! __export_flags_service_impl {
    ($ty:ident) => {
        self::export!($ty with_types_in self);
    };
    ($ty:ident with_types_in $($path_to_types_root:tt)*) => {
        $($path_to_types_root)*::
        exports::golem::it::api::__export_golem_it_api_cabi!($ty with_types_in
        $($path_to_types_root)*:: exports::golem::it::api);
    };
}
#[doc(inline)]
pub(crate) use __export_flags_service_impl as export;
#[cfg(target_arch = "wasm32")]
#[unsafe(
    link_section = "component-type:wit-bindgen:0.41.0:golem:it:flags-service:encoded world"
)]
#[doc(hidden)]
#[allow(clippy::octal_escapes)]
pub static __WIT_BINDGEN_COMPONENT_TYPE: [u8; 309] = *b"\
\0asm\x0d\0\x01\0\0\x19\x16wit-component-encoding\x04\0\x07\xb1\x01\x01A\x02\x01\
A\x02\x01B\x09\x01n\x04\x04read\x05write\x04exec\x05close\x04\0\x0bpermissions\x03\
\0\0\x01r\x02\x04names\x0bpermissions\x01\x04\0\x04task\x03\0\x02\x01@\x01\x05in\
put\x03\0\x03\x04\0\x0bcreate-task\x01\x04\x01p\x03\x01@\0\0\x05\x04\0\x09get-ta\
sks\x01\x06\x04\0\x0cgolem:it/api\x05\0\x04\0\x16golem:it/flags-service\x04\0\x0b\
\x13\x01\0\x0dflags-service\x03\0\0\0G\x09producers\x01\x0cprocessed-by\x02\x0dw\
it-component\x070.227.1\x10wit-bindgen-rust\x060.41.0";
#[inline(never)]
#[doc(hidden)]
pub fn __link_custom_section_describing_imports() {
    wit_bindgen_rt::maybe_link_cabi_realloc();
}
