//! QR decomposition.

use crate::{blas, lapack, Real, Scalar};
use ndarray::{s, Array1, Array2, ArrayBase, Data, DataMut, Ix2};
use std::fmt;
use std::ops::{Div, MulAssign};

/// QR decomposition factors.
#[derive(Debug)]
pub struct QRFactorized<A, S>
where
    A: fmt::Debug,
    S: Data<Elem = A>,
{
    qr: ArrayBase<S, Ix2>,
    tau: Array1<A>,
}

impl<A, S> QRFactorized<A, S>
where
    A: Scalar + fmt::Debug,
    S: Data<Elem = A>,
{
    /// Returns *Q* of QR decomposition.
    pub fn q(&self) -> Array2<A> {
        let mut q = Array2::<A>::zeros((self.qr.nrows(), self.qr.nrows()));
        for j in 0..self.tau.len() {
            for i in 0..self.qr.nrows() {
                q[(i, j)] = self.qr[(i, j)];
            }
        }
        for j in self.tau.len()..q.ncols() {
            for l in 0..q.nrows() {
                q[(l, j)] = A::zero();
            }
            q[(j, j)] = A::one();
        }

        for i in (0..self.tau.len()).rev() {
            if i < q.ncols() - 1 {
                q[(i, i)] = A::one();
                let v = q.column(i).slice(s![i..]).to_owned();
                lapack::larf::left(&v, self.tau[i], &mut q.slice_mut(s![i.., i + 1..]));
            }
            if i < q.nrows() - 1 {
                blas::scal(-self.tau[i], &mut q.column_mut(i).slice_mut(s![i + 1..]));
            }
            q[(i, i)] = A::one() - self.tau[i];

            if i > 0 {
                for l in 0..i {
                    q[(l, i)] = A::zero();
                }
            }
        }
        q
    }

    /// Returns *R* of QR decomposition.
    pub fn r(&self) -> Array2<A> {
        let mut r = self.qr.to_owned();
        for i in 1..r.nrows() {
            for j in 0..i {
                r[(i, j)] = A::zero();
            }
        }
        r
    }
}

impl<A, S> From<ArrayBase<S, Ix2>> for QRFactorized<A, S>
where
    A: Scalar + Div<<A as Scalar>::Real, Output = A> + MulAssign<<A as Scalar>::Real>,
    A::Real: Real,
    S: DataMut<Elem = A>,
{
    /// Converts a matrix into the QR-factorized form, *Q* * *R*.
    fn from(mut a: ArrayBase<S, Ix2>) -> Self {
        let tau = lapack::geqrf(&mut a);
        QRFactorized { qr: a, tau }
    }
}

#[cfg(test)]
mod tests {
    use approx::AbsDiffEq;
    use ndarray::{arr1, arr2};

    #[test]
    fn square() {
        let a = arr2(&[
            [1_f64, 2_f64, 3_f64],
            [2_f64, 2_f64, 1_f64],
            [3_f64, 1_f64, 2_f64],
        ]);
        let qr = super::QRFactorized::from(a);
        assert!(qr.qr.abs_diff_eq(
            &arr2(&[
                [-3.74165739, -2.40535118, -2.93987366],
                [0.42179344, -1.79284291, -1.6334791],
                [0.63269017, -0.9237749, -1.63978318]
            ]),
            1e-6
        ));
        assert!(qr
            .tau
            .abs_diff_eq(&arr1(&[1.26726124, 1.07912113, 0.]), 1e-6));
        let q = qr.q();
        assert_eq!(q.shape(), &[3, 3]);
        assert!(q.abs_diff_eq(
            &arr2(&[
                [-0.26726124, -0.75697812, -0.59628479],
                [-0.53452248, -0.39840954, 0.74535599],
                [-0.80178373, 0.5179324, -0.2981424]
            ]),
            1e-6
        ));
        let r = qr.r();
        assert_eq!(r.shape(), &[3, 3]);
        assert!(r.abs_diff_eq(
            &arr2(&[
                [-3.74165739, -2.40535118, -2.93987366],
                [0., -1.79284291, -1.6334791],
                [0., 0., -1.63978318]
            ]),
            1e-6
        ));
    }

    #[test]
    fn wide() {
        let a = arr2(&[
            [1_f32, 2_f32, 3_f32, 1_f32],
            [2_f32, 2_f32, 1_f32, 3_f32],
            [3_f32, 1_f32, 2_f32, 2_f32],
        ]);
        let qr = super::QRFactorized::from(a);
        let q = qr.q();
        assert_eq!(q.shape(), &[3, 3]);
        assert!(q.abs_diff_eq(
            &arr2(&[
                [-0.26726124, -0.75697812, -0.59628479],
                [-0.53452248, -0.39840954, 0.74535599],
                [-0.80178373, 0.5179324, -0.2981424]
            ]),
            1e-6
        ));
        let r = qr.r();
        assert_eq!(r.shape(), &[3, 4]);
        assert!(r.abs_diff_eq(
            &arr2(&[
                [-3.74165739, -2.40535118, -2.93987366, -3.47439614],
                [0., -1.79284291, -1.6334791, -0.91634193],
                [0., 0., -1.63978318, 1.04349839]
            ]),
            1e-6
        ));
    }

    #[test]
    fn tall() {
        let a = arr2(&[
            [1_f64, 2_f64, 3_f64],
            [2_f64, 2_f64, 1_f64],
            [3_f64, 1_f64, 2_f64],
            [2_f64, 3_f64, 3_f64],
        ]);
        let qr = super::QRFactorized::from(a);
        let q = qr.q();
        assert_eq!(q.shape(), &[4, 4]);
        assert!(q.abs_diff_eq(
            &arr2(&[
                [-0.23570226, 0.49746834, -0.62737462, -0.55079106],
                [-0.47140452, 0.14213381, 0.71963559, -0.48959205],
                [-0.70710678, -0.63960215, -0.29523511, 0.06119901],
                [-0.47140452, 0.56853524, 0.03690439, 0.67318907]
            ]),
            1e-6
        ));
        let r = qr.r();
        assert_eq!(r.shape(), &[4, 3]);
        assert!(r.abs_diff_eq(
            &arr2(&[
                [-4.24264069, -3.53553391, -4.00693843],
                [0., 2.34520788, 2.06094026],
                [0., 0., -1.64224532],
                [0., 0., 0.]
            ]),
            1e-6
        ));
    }
}
