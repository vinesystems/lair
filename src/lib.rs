//! Lair implements linear algebra routines in pure Rust. It uses [ndarray] as
//! its matrix representation.
//!
//! [ndarray]: https://docs.rs/ndarray/

mod blas;
pub mod decomposition;
// TODO: pub mod eigen;
pub mod equation;
mod lapack;
pub mod matrix;
mod scalar;

pub use scalar::{Real, Scalar};

#[cfg(feature = "bench-lapack")]
#[doc(hidden)]
pub mod bench_lapack {
    pub use crate::lapack::gebrd::tall as gebrd_tall;
    pub use crate::lapack::*;
}

/// An error which can be returned when a function argument is invalid.
#[derive(Debug, thiserror::Error)]
pub enum InvalidInput {
    #[error("shape error: {0}")]
    Shape(String),
    #[error("value error: {0}")]
    Value(String),
}
