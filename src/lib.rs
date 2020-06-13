//! Lair implements linear algebra routines in pure Rust. It uses [ndarray] as
//! its matrix representation.
//!
//! [ndarray]: https://docs.rs/ndarray/

pub mod decomposition;
// TODO: pub mod eigen;
pub mod equation;
pub(crate) mod lapack;
pub mod matrix;

/// An error which can be returned when a function argument is invalid.
#[derive(Debug, thiserror::Error)]
pub enum InvalidInput {
    #[error("shape error: {0}")]
    Shape(String),
    #[error("value error: {0}")]
    Value(String),
}
