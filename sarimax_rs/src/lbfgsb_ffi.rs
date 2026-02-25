//! Raw FFI bindings to the thread-safe L-BFGS-B C library.
//!
//! This replaces the `lbfgsb` crate's bindgen-generated bindings with
//! hand-written declarations matching lbfgsb.h.  The C source files
//! in `lbfgsb_c/` are compiled by build.rs with `_Thread_local` on all
//! function-local static variables, making `setulb` safe to call from
//! multiple threads concurrently.

#![allow(nonstandard_style, dead_code)]

pub type integer = i64;
pub type logical = i64;

extern "C" {
    pub fn setulb(
        n: *const integer,
        m: *const integer,
        x: *mut f64,
        l: *const f64,
        u: *const f64,
        nbd: *const integer,
        f: *mut f64,
        g: *mut f64,
        factr: *const f64,
        pgtol: *const f64,
        wa: *mut f64,
        iwa: *mut integer,
        task: *mut integer,
        iprint: *const integer,
        csave: *mut integer,
        lsave: *mut logical,
        isave: *mut integer,
        dsave: *mut f64,
    ) -> std::os::raw::c_int;
}

// Task constants from lbfgsb.h
pub const START: i64 = 1;
pub const NEW_X: i64 = 2;

pub const FG: i64 = 10;
pub const FG_END: i64 = 15;

pub const CONVERGENCE: i64 = 20;

#[inline]
pub fn is_fg(task: i64) -> bool {
    task >= FG && task <= FG_END
}
