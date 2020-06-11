//! Lair implements linear algebra routines in pure Rust. It uses [ndarray] as
//! its matrix representation.
//!
//! [ndarray]: https://docs.rs/ndarray/

pub(crate) mod lapack;
pub mod matrices;

use ndarray::{Array1, ArrayBase, Data, Ix1, Ix2, NdFloat};
use std::error::Error;
use std::fmt;

/// An error which can be returned when a function argument is invalid.
#[derive(Debug)]
pub struct InvalidInput {
    msg: String,
}

impl From<String> for InvalidInput {
    fn from(msg: String) -> Self {
        Self { msg }
    }
}

impl fmt::Display for InvalidInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid input: {}", self.msg)
    }
}

impl Error for InvalidInput {}

pub fn solve<A, SA, SB>(a: &ArrayBase<SA, Ix2>, b: &ArrayBase<SB, Ix1>) -> Array1<A>
where
    A: NdFloat,
    SA: Data<Elem = A>,
    SB: Data<Elem = A>,
{
    let mut a = a.to_owned();
    let p = lapack::getrf(&mut a);
    lapack::getrs(&a, &p, b)
}
