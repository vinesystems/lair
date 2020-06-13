//! LU decomposition.

use crate::lapack;
use ndarray::{Array1, Array2, ArrayBase, Data, DataMut, Ix1, Ix2, NdFloat};
use std::cmp::min;
use std::convert::TryFrom;
use std::fmt;

/// LU decomposition factors.
#[derive(Debug)]
pub struct LUFactorized<A, S>
where
    A: fmt::Debug,
    S: Data<Elem = A>,
{
    lu: ArrayBase<S, Ix2>,
    pivots: Vec<usize>,
}

impl<A, S> LUFactorized<A, S>
where
    A: NdFloat + fmt::Debug,
    S: Data<Elem = A>,
{
    /// Returns the permutation matrix *P* of LU decomposition.
    pub fn p(&self) -> Array2<A> {
        let mut p = Array2::zeros((self.lu.nrows(), self.lu.nrows()));
        for (i, pivot) in self.pivots.iter().enumerate() {
            p[(*pivot, i)] = A::one();
        }
        p
    }

    /// Returns *L* of LU decomposition.
    pub fn l(&self) -> Array2<A> {
        let rank = min(self.lu.nrows(), self.lu.ncols());
        let mut l = Array2::zeros((self.lu.nrows(), rank));
        for i in 0..self.lu.nrows() {
            for j in 0..i {
                l[(i, j)] = self.lu[(i, j)];
            }
            if i < rank {
                l[(i, i)] = A::one();
                for j in i + 1..rank {
                    l[(i, j)] = A::zero();
                }
            }
        }
        l
    }

    /// Returns *U* of LU decomposition.
    pub fn u(&self) -> Array2<A> {
        let rank = min(self.lu.nrows(), self.lu.ncols());
        let mut u = Array2::zeros((rank, self.lu.ncols()));
        for i in 0..rank {
            for j in 0..i {
                u[(i, j)] = A::zero();
            }
            for j in i..self.lu.ncols() {
                u[(i, j)] = self.lu[(i, j)];
            }
        }
        u
    }

    /// Solves the system of equations `P * L * U * x = b` for `x`.
    pub fn solve<SB>(&self, b: &ArrayBase<SB, Ix1>) -> Array1<A>
    where
        SB: Data<Elem = A>,
    {
        lapack::getrs(&self.lu, &self.pivots, b)
    }
}

impl<A, S> TryFrom<ArrayBase<S, Ix2>> for LUFactorized<A, S>
where
    A: NdFloat + fmt::Debug,
    S: DataMut<Elem = A>,
{
    type Error = Singular;

    /// Converts a matrix into the LU-factorized form, *P* * *L* * *U*.
    fn try_from(mut a: ArrayBase<S, Ix2>) -> Result<Self, Self::Error> {
        let pivots = match lapack::getrf(&mut a) {
            Ok(ipiv) => ipiv,
            Err(_) => return Err(Singular),
        };
        Ok(LUFactorized { lu: a, pivots })
    }
}

/// Returned as an error when the input matrix is singular while expected
/// non-sigular.
#[derive(Debug)]
pub struct Singular;

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use ndarray::arr2;
    use std::convert::TryFrom;

    #[test]
    fn square() {
        let a = arr2(&[
            [1_f32, 2_f32, 3_f32],
            [2_f32, 2_f32, 1_f32],
            [3_f32, 1_f32, 2_f32],
        ]);
        let lu = super::LUFactorized::try_from(a).expect("non-singular");
        let p = lu.p();
        assert_eq!(p[(0, 1)], 1.);
        assert_eq!(p[(1, 2)], 1.);
        assert_eq!(p[(2, 0)], 1.);
        let l = lu.l();
        assert_relative_eq!(l[(0, 0)], 1., max_relative = 1e-6);
        assert_relative_eq!(l[(1, 0)], 0.33333333, max_relative = 1e-6);
        assert_relative_eq!(l[(2, 0)], 0.66666666, max_relative = 1e-6);
        let u = lu.u();
        assert_relative_eq!(u[(0, 2)], 2., max_relative = 1e-6);
        assert_relative_eq!(u[(1, 2)], 2.33333333, max_relative = 1e-6);
        assert_relative_eq!(u[(2, 2)], -2.2, max_relative = 1e-6);
    }

    #[test]
    fn wide() {
        let a = arr2(&[
            [1_f32, 2_f32, 3_f32, 1_f32],
            [2_f32, 2_f32, 1_f32, 3_f32],
            [3_f32, 1_f32, 2_f32, 2_f32],
        ]);
        let lu = super::LUFactorized::try_from(a).expect("non-singular");
        let p = lu.p();
        assert_eq!(p.shape(), &[3, 3]);
        assert_eq!(p[(0, 1)], 1.);
        assert_eq!(p[(1, 2)], 1.);
        assert_eq!(p[(2, 0)], 1.);
        let l = lu.l();
        assert_eq!(l.shape(), &[3, 3]);
        assert_relative_eq!(l[(0, 0)], 1., max_relative = 1e-6);
        assert_relative_eq!(l[(1, 0)], 0.33333333, max_relative = 1e-6);
        assert_relative_eq!(l[(2, 0)], 0.66666666, max_relative = 1e-6);
        let u = lu.u();
        assert_eq!(u.shape(), &[3, 4]);
        assert_relative_eq!(u[(0, 2)], 2., max_relative = 1e-6);
        assert_relative_eq!(u[(1, 2)], 2.33333333, max_relative = 1e-6);
        assert_relative_eq!(u[(2, 2)], -2.2, max_relative = 1e-6);
    }

    #[test]
    fn tall() {
        let a = arr2(&[
            [1_f32, 2_f32, 3_f32],
            [2_f32, 2_f32, 1_f32],
            [3_f32, 1_f32, 2_f32],
            [2_f32, 3_f32, 3_f32],
        ]);
        let lu = super::LUFactorized::try_from(a).expect("non-singular");
        let p = lu.p();
        assert_eq!(p.shape(), &[4, 4]);
        assert_eq!(p[(0, 3)], 1.);
        assert_eq!(p[(1, 2)], 1.);
        assert_eq!(p[(2, 0)], 1.);
        assert_eq!(p[(3, 1)], 1.);
        let l = lu.l();
        assert_eq!(l.shape(), &[4, 3]);
        assert_relative_eq!(l[(0, 0)], 1., max_relative = 1e-6);
        assert_relative_eq!(l[(1, 0)], 0.66666666, max_relative = 1e-6);
        assert_relative_eq!(l[(2, 0)], 0.66666666, max_relative = 1e-6);
        assert_relative_eq!(l[(3, 0)], 0.33333333, max_relative = 1e-6);
        let u = lu.u();
        assert_eq!(u.shape(), &[3, 3]);
        assert_relative_eq!(u[(0, 2)], 2., max_relative = 1e-6);
        assert_relative_eq!(u[(1, 2)], 1.66666666, max_relative = 1e-6);
        assert_relative_eq!(u[(2, 2)], -1.28571429, max_relative = 1e-6);
    }
}