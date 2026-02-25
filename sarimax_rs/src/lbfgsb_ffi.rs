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

// Task constants from lbfgsb.h — full set for proper termination handling.
pub const START: i64 = 1;
pub const NEW_X: i64 = 2;
pub const ABNORMAL: i64 = 3;

pub const FG: i64 = 10;
pub const FG_END: i64 = 15;

pub const CONVERGENCE: i64 = 20;
pub const CONVERGENCE_END: i64 = 25;

pub const STOP: i64 = 30;
pub const STOP_END: i64 = 40;

pub const WARNING: i64 = 100;
pub const WARNING_END: i64 = 110;

pub const ERROR: i64 = 200;
pub const ERROR_END: i64 = 240;

#[inline]
pub fn is_fg(task: i64) -> bool {
    task >= FG && task <= FG_END
}

#[inline]
pub fn is_converged(task: i64) -> bool {
    task >= CONVERGENCE && task <= CONVERGENCE_END
}

#[inline]
pub fn is_warning(task: i64) -> bool {
    task >= WARNING && task <= WARNING_END
}

#[inline]
pub fn is_error(task: i64) -> bool {
    task >= ERROR && task <= ERROR_END
}
