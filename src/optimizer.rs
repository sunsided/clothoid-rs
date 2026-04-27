//! Path evaluation and optimization utilities for clothoid fitting.
//!
//! This module provides:
//! - RK4 integration for clothoid arcs with linearly varying curvature
//! - Path evaluation from flat parameter vectors
//! - A Nelder-Mead simplex optimizer for derivative-free minimization
//! - A linear congruential generator (LCG) for reproducible random initialization

pub mod angle;
pub mod integrate;
pub mod objective;
pub mod path;
pub mod pose;
pub mod state;

pub use angle::angle_diff;
pub use integrate::{integrate_clothoid, rk4_step};
pub use objective::{
    PlanObjective, SymmetryMode, compute_end_errors, compute_error, is_symmetric_task,
};
pub use path::{PathSegment, SegmentKind, eval_path, eval_path_segmented};
pub use pose::Pose;
pub use state::ClothoidState;

#[cfg(any(feature = "nelder-mead", feature = "cma-es"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "nelder-mead", feature = "cma-es"))))]
pub mod solver;
#[cfg(any(feature = "nelder-mead", feature = "cma-es"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "nelder-mead", feature = "cma-es"))))]
pub use solver::Optimizer;

#[cfg(feature = "nelder-mead")]
#[cfg_attr(docsrs, doc(cfg(feature = "nelder-mead")))]
pub mod nelder_mead;
#[cfg(feature = "nelder-mead")]
#[cfg_attr(docsrs, doc(cfg(feature = "nelder-mead")))]
pub use nelder_mead::{NelderMead, nelder_mead};

#[cfg(feature = "cma-es")]
#[cfg_attr(docsrs, doc(cfg(feature = "cma-es")))]
pub mod cmaes;
#[cfg(feature = "cma-es")]
#[cfg_attr(docsrs, doc(cfg(feature = "cma-es")))]
pub mod linalg;
pub mod rng;

#[cfg(feature = "cma-es")]
#[cfg_attr(docsrs, doc(cfg(feature = "cma-es")))]
pub use cmaes::CmaEs;
#[cfg(feature = "cma-es")]
#[cfg_attr(docsrs, doc(cfg(feature = "cma-es")))]
pub use linalg::jacobi_eigen;
#[cfg(feature = "cma-es")]
#[cfg_attr(docsrs, doc(cfg(feature = "cma-es")))]
pub use rng::randn;
pub use rng::{DEFAULT_RNG_SEED, Lcg};
