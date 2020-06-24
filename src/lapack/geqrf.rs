use crate::{lapack, Real, Scalar};
use ndarray::{s, Array1, ArrayBase, DataMut, Ix2};
use std::cmp;

/// Computes the QR factorization of a matrix.
pub fn geqrf<A, S>(a: &mut ArrayBase<S, Ix2>) -> Array1<A>
where
    A: Scalar,
    A::Real: Real,
    S: DataMut<Elem = A>,
{
    let min_dim = cmp::min(a.nrows(), a.ncols());
    let mut tau = Array1::<A>::zeros(min_dim);
    for i in 0..min_dim {
        let bottom = a.nrows();
        let (beta, _, t) = lapack::larfg(a[(i, i)], a.column_mut(i).slice_mut(s![i + 1..bottom]));
        tau[i] = t;
        if i < a.ncols() {
            a[(i, i)] = A::one();
            let v = a.column(i).slice(s![i..]).to_owned();
            lapack::larf::left(&v, t.conj(), &mut a.slice_mut(s![i.., i + 1..]));
            a[(i, i)] = A::from_real(beta);
        }
    }
    tau
}

#[cfg(test)]
mod test {
    use approx::{assert_abs_diff_eq, AbsDiffEq};
    use ndarray::{arr1, arr2};
    use num_complex::Complex64;

    #[test]
    fn square_real_smallest() {
        let a = arr2(&[[2_f64]]);
        let mut qr = a.clone();
        assert_eq!(qr.shape(), &[1, 1]);
        let tau = super::geqrf(&mut qr);
        assert_eq!(tau.shape(), &[1]);
        assert!(qr.abs_diff_eq(&arr2(&[[2_f64]]), 1e-8));
        assert!(tau.abs_diff_eq(&arr1(&[0.]), 1e-8));
    }

    #[test]
    fn square_real() {
        let a = arr2(&[
            [1_f64, 2_f64, 4_f64],
            [0_f64, 0_f64, 5_f64],
            [0_f64, 3_f64, 6_f64],
        ]);
        let mut qr = a.clone();
        assert_eq!(qr.shape(), &[3, 3]);
        let tau = super::geqrf(&mut qr);
        assert_eq!(tau.shape(), &[3]);
        assert!(qr.abs_diff_eq(
            &arr2(&[
                [1_f64, 2_f64, 4_f64],
                [0_f64, -3_f64, -6_f64],
                [0_f64, 1_f64, -5_f64],
            ]),
            1e-8
        ));
        assert!(tau.abs_diff_eq(&arr1(&[0., 1., 0.]), 1e-8));
    }

    #[test]
    fn square_complex() {
        let a = arr2(&[
            [Complex64::new(1., 1.), Complex64::new(2., -1.)],
            [Complex64::new(3., 1.), Complex64::new(4., -1.)],
        ]);
        let mut qr = a.clone();
        assert_eq!(qr.shape(), &[2, 2]);
        let tau = super::geqrf(&mut qr);
        assert_eq!(tau.shape(), &[2]);
        assert_abs_diff_eq!(qr[(0, 0)].re, -3.46410162, epsilon = 1e-6);
        assert_abs_diff_eq!(qr[(0, 0)].im, 0., epsilon = 1e-6);
        assert_abs_diff_eq!(qr[(0, 1)].re, -3.46410162, epsilon = 1e-6);
        assert_abs_diff_eq!(qr[(0, 1)].im, 2.88675135, epsilon = 1e-6);
        assert_abs_diff_eq!(qr[(1, 0)].re, 0.68769902, epsilon = 1e-6);
        assert_abs_diff_eq!(qr[(1, 0)].im, 0.0699583, epsilon = 1e-6);
        assert_abs_diff_eq!(qr[(1, 1)].re, 1.29099445, epsilon = 1e-6);
        assert_abs_diff_eq!(qr[(1, 1)].im, 0., epsilon = 1e-6);
        assert_abs_diff_eq!(tau[0].re, 1.28867513, epsilon = 1e-6);
        assert_abs_diff_eq!(tau[0].im, 0.28867513, epsilon = 1e-6);
        assert_abs_diff_eq!(tau[1].re, 1.02290316, epsilon = 1e-6);
        assert_abs_diff_eq!(tau[1].im, -0.99973769, epsilon = 1e-6);
    }

    #[test]
    fn tall() {
        let a = arr2(&[
            [1_f64, 2_f64, 3_f64],
            [2_f64, 2_f64, 1_f64],
            [3_f64, 1_f64, 2_f64],
            [2_f64, 3_f64, 3_f64],
        ]);
        let mut qr = a.clone();
        let tau = super::geqrf(&mut qr);
        assert!(qr.abs_diff_eq(
            &arr2(&[
                [-4.24264069, -3.53553391, -4.00693843],
                [0.38148714, 2.34520788, 2.06094026],
                [0.57223071, 0.88223561, -1.64224532],
                [0.38148714, -0.36153262, -0.34951992]
            ]),
            1e-6
        ));
        assert!(tau.abs_diff_eq(&arr1(&[1.23570226, 1.04764396, 1.7822704]), 1e-6));
    }

    #[test]
    fn wide() {
        let a = arr2(&[
            [1_f32, 2_f32, 3_f32, 1_f32],
            [2_f32, 2_f32, 1_f32, 3_f32],
            [3_f32, 1_f32, 2_f32, 2_f32],
        ]);
        let mut qr = a.clone();
        let tau = super::geqrf(&mut qr);
        assert!(qr.abs_diff_eq(
            &arr2(&[
                [-3.74165739, -2.40535118, -2.93987366, -3.47439614],
                [0.42179344, -1.79284291, -1.6334791, -0.91634193],
                [0.63269017, -0.9237749, -1.63978318, 1.04349839]
            ]),
            1e-6
        ));
        assert!(tau.abs_diff_eq(&arr1(&[1.26726124, 1.07912113, 0.]), 1e-6));
    }
}
