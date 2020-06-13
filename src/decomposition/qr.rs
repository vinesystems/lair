//! QR decomposition.

use crate::lapack;
use ndarray::{Array2, ArrayBase, Data, DataMut, Ix2, NdFloat};
use std::fmt;
use std::iter::Sum;

/// QR decomposition factors.
#[derive(Debug)]
pub struct QRFactorized<A, S>
where
    A: fmt::Debug,
    S: Data<Elem = A>,
{
    q: ArrayBase<S, Ix2>,
    r: Array2<A>,
}

impl<A, S> QRFactorized<A, S>
where
    A: Clone + fmt::Debug,
    S: Data<Elem = A>,
{
    /// Returns *Q* of QR decomposition.
    pub fn q(&self) -> Array2<A> {
        self.q.to_owned()
    }

    /// Returns *R* of QR decomposition.
    pub fn r(&self) -> Array2<A> {
        self.r.clone()
    }
}

impl<A, S> From<ArrayBase<S, Ix2>> for QRFactorized<A, S>
where
    A: NdFloat + Sum + fmt::Debug,
    S: DataMut<Elem = A>,
{
    /// Converts a matrix into the QR-factorized form, *Q* * *R*.
    fn from(mut a: ArrayBase<S, Ix2>) -> Self {
        let r = lapack::geqrf(&mut a);
        QRFactorized { q: a, r }
    }
}
