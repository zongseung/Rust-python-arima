//! High-level Rust wrapper around the L-BFGS-B C library.
//!
//! Provides `LbfgsbState`, `LbfgsbProblem`, and `LbfgsbParameter`
//! (previously from the `lbfgsb` crate).  Uses our thread-safe
//! `lbfgsb_ffi` bindings instead of upstream.

use crate::lbfgsb_ffi::{self, integer, logical};
use anyhow::Result;

/// L-BFGS-B algorithm parameters.
pub struct LbfgsbParameter {
    /// Maximum number of variable metric corrections (memory size).
    pub m: usize,
    /// Function-value tolerance: iteration stops when
    /// `(f^k - f^{k+1})/max(|f^k|, |f^{k+1}|, 1) <= factr * epsmch`.
    pub factr: f64,
    /// Projected gradient tolerance.
    pub pgtol: f64,
    /// Verbosity (-1 = silent).
    pub iprint: i64,
}

impl Default for LbfgsbParameter {
    fn default() -> Self {
        Self {
            m: 5,
            factr: 1e7,
            pgtol: 1e-5,
            iprint: -1,
        }
    }
}

pub struct LbfgsbProblem<E>
where
    E: FnMut(&[f64], &mut [f64]) -> Result<f64>,
{
    x: Vec<f64>,
    g: Vec<f64>,
    f: f64,
    l: Vec<f64>,
    u: Vec<f64>,
    nbd: Vec<i64>,
    eval_fn: E,
}

impl<E> LbfgsbProblem<E>
where
    E: FnMut(&[f64], &mut [f64]) -> Result<f64>,
{
    pub fn build(x: Vec<f64>, eval_fn: E) -> Self {
        let n = x.len();
        Self {
            x,
            g: vec![0.0; n],
            f: 0.0,
            l: vec![0.0; n],
            u: vec![0.0; n],
            nbd: vec![0; n],
            eval_fn,
        }
    }

    pub fn set_bounds<B>(&mut self, bounds: B)
    where
        B: IntoIterator<Item = (Option<f64>, Option<f64>)>,
    {
        for (i, b) in bounds.into_iter().enumerate() {
            match b {
                (Some(l), Some(u)) => {
                    self.l[i] = l;
                    self.u[i] = u;
                    self.nbd[i] = 2;
                }
                (None, None) => {
                    self.nbd[i] = 0;
                }
                (Some(l), None) => {
                    self.l[i] = l;
                    self.nbd[i] = 1;
                }
                (None, Some(u)) => {
                    self.u[i] = u;
                    self.nbd[i] = 3;
                }
            }
        }
    }
}

pub struct LbfgsbState<E>
where
    E: FnMut(&[f64], &mut [f64]) -> Result<f64>,
{
    problem: LbfgsbProblem<E>,
    param: LbfgsbParameter,
    wa: Vec<f64>,
    iwa: Vec<i64>,
    csave: [integer; 60],
    dsave: [f64; 29],
    isave: [integer; 44],
    lsave: [logical; 4],
    task: integer,
}

impl<E> LbfgsbState<E>
where
    E: FnMut(&[f64], &mut [f64]) -> Result<f64>,
{
    pub fn new(problem: LbfgsbProblem<E>, param: LbfgsbParameter) -> Self {
        let n = problem.x.len();
        let m = param.m;
        let wa = vec![0.0; 2 * m * n + 5 * n + 11 * m * m + 8 * m];
        let iwa = vec![0; 3 * n];
        Self {
            csave: [0; 60],
            dsave: [0.0; 29],
            isave: [0; 44],
            lsave: [0; 4],
            task: lbfgsb_ffi::START,
            problem,
            param,
            wa,
            iwa,
        }
    }

    /// Run the L-BFGS-B optimizer loop.
    ///
    /// Returns the final Fortran task code on success:
    /// - CONVERGENCE (20-25): solver converged normally
    /// - STOP (30-40): solver stopped (e.g. iteration/CPU limit)
    /// - WARNING (100-110): solver terminated with a warning
    ///
    /// Returns `Err` for ERROR (200+) or ABNORMAL (3) terminations.
    pub fn minimize(&mut self) -> Result<i64> {
        let f = &mut self.problem.f;
        let x = &mut self.problem.x;
        let g = &mut self.problem.g;
        let l = &self.problem.l;
        let u = &self.problem.u;
        let nbd = &self.problem.nbd;
        let param = &self.param;
        let n = x.len();
        let m = param.m;

        loop {
            unsafe {
                lbfgsb_ffi::setulb(
                    &(n as i64),
                    &(m as i64),
                    x.as_mut_ptr(),
                    l.as_ptr(),
                    u.as_ptr(),
                    nbd.as_ptr(),
                    f,
                    g.as_mut_ptr(),
                    &param.factr,
                    &param.pgtol,
                    self.wa.as_mut_ptr(),
                    self.iwa.as_mut_ptr(),
                    &mut self.task,
                    &param.iprint,
                    self.csave.as_mut_ptr(),
                    self.lsave.as_mut_ptr(),
                    self.isave.as_mut_ptr(),
                    self.dsave.as_mut_ptr(),
                );
            }
            if lbfgsb_ffi::is_fg(self.task) {
                *f = (self.problem.eval_fn)(x, g)?;
            } else if self.task == lbfgsb_ffi::NEW_X {
                // Continue iteration.
            } else {
                break;
            }
        }

        // Classify final task code
        if lbfgsb_ffi::is_error(self.task) || self.task == lbfgsb_ffi::ABNORMAL {
            return Err(anyhow::anyhow!(
                "L-BFGS-B terminated with error (task code {})",
                self.task
            ));
        }

        Ok(self.task)
    }

    pub fn fx(&self) -> f64 {
        self.problem.f
    }

    pub fn x(&self) -> &[f64] {
        &self.problem.x
    }
}
