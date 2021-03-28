//! LU decomposition.

use crate::{lapack, InvalidInput, Real, Scalar};
use ndarray::{Array1, Array2, ArrayBase, Data, DataMut, Ix1, Ix2};
use std::cmp::min;
use std::fmt;

/// LU decomposition factors.
#[derive(Debug)]
pub struct Factorized<A, S>
where
    A: fmt::Debug,
    S: Data<Elem = A>,
{
    lu: ArrayBase<S, Ix2>,
    pivots: Vec<usize>,
    singular: Option<usize>,
}

impl<A, S> Factorized<A, S>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    /// Returns the permutation matrix *P* of LU decomposition.
    pub fn p(&self) -> Array2<A> {
        let permutation = {
            let mut permutation = (0..self.lu.nrows()).collect::<Vec<_>>();
            unsafe { lapack::laswp(1, permutation.as_mut_ptr(), 1, 1, 0, &self.pivots) };
            permutation
        };
        let mut p = Array2::zeros((self.lu.nrows(), self.lu.nrows()));
        for (i, pivot) in permutation.iter().enumerate() {
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

    /// Returns `true` if the matrix is singular.
    pub fn is_singular(&self) -> bool {
        self.singular.is_some()
    }

    /// Solves the system of equations `P * L * U * x = b` for `x`.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidInput::Shape`] if the number of elements in `b` does
    /// not match with the number of rows in *L*.
    ///
    /// [`InvalidInput::Shape`]: ../enum.InvalidInput.html#variant.Shape
    pub fn solve<SB>(&self, b: &ArrayBase<SB, Ix1>) -> Result<Array1<A>, InvalidInput>
    where
        SB: Data<Elem = A>,
    {
        if b.len() != self.lu.nrows() {
            return Err(InvalidInput::Shape(format!(
                "b must have {} elements",
                self.lu.nrows()
            )));
        }
        Ok(lapack::getrs(&self.lu, &self.pivots, b))
    }
}

impl<A, S> From<ArrayBase<S, Ix2>> for Factorized<A, S>
where
    A: Scalar,
    A::Real: Real,
    S: DataMut<Elem = A>,
{
    /// Converts a matrix into the LU-factorized form, *P* * *L* * *U*.
    fn from(mut a: ArrayBase<S, Ix2>) -> Self {
        let (pivots, singular) = lapack::getrf(a.view_mut());
        Factorized {
            lu: a,
            pivots,
            singular,
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::arr2;

    #[test]
    fn square() {
        let a = arr2(&[
            [1_f32, 2_f32, 3_f32],
            [2_f32, 2_f32, 1_f32],
            [3_f32, 1_f32, 2_f32],
        ]);
        let lu = super::Factorized::from(a);
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
        let lu = super::Factorized::from(a);
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
        let lu = super::Factorized::from(a);
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
