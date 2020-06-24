//! Lair implements linear algebra routines in pure Rust. It uses [ndarray] as
//! its matrix representation.
//!
//! [ndarray]: https://docs.rs/ndarray/

pub(crate) mod blas;
pub mod decomposition;
// TODO: pub mod eigen;
pub mod equation;
pub(crate) mod lapack;
pub mod matrix;

/// A trait for real and complex numbers.
pub use cauchy::Scalar;

/// A trait for real numbers.
pub trait Real: num_traits::Float {
    /// Returns a number composed of the magnitude of `self` and the sign of
    /// `sign`.
    fn copysign(self, sign: Self) -> Self;

    /// Takes the reciprocal (inverse) of a number.
    fn recip(self) -> Self;

    /// Relative machine precision.
    fn eps() -> Self;

    /// Safe minimum, such that its reciprocal does not overflow.
    fn sfmin() -> Self;
}

impl Real for f64 {
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        self.copysign(sign)
    }

    #[inline]
    fn recip(self) -> Self {
        self.recip()
    }

    #[inline]
    fn eps() -> Self {
        1.110_223_024_625_156_5e-16
    }

    #[inline]
    fn sfmin() -> Self {
        2.225_073_858_507_201_4e-_308
    }
}

impl Real for f32 {
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        self.copysign(sign)
    }

    #[inline]
    fn recip(self) -> Self {
        self.recip()
    }

    #[inline]
    fn eps() -> Self {
        5.960_464_5e-8
    }

    #[inline]
    fn sfmin() -> Self {
        1.175_494_4e-38
    }
}

/// An error which can be returned when a function argument is invalid.
#[derive(Debug, thiserror::Error)]
pub enum InvalidInput {
    #[error("shape error: {0}")]
    Shape(String),
    #[error("value error: {0}")]
    Value(String),
}
