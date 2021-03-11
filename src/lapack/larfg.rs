use crate::{blas, lapack, Float, Real, Scalar};
use ndarray::{ArrayBase, DataMut, Ix1};
use std::ops::{Div, MulAssign};

/// Generates an elementary reflector (Householder matrix).
pub fn larfg<A, S>(mut alpha: A, mut x: ArrayBase<S, Ix1>) -> (A::Real, ArrayBase<S, Ix1>, A)
where
    A: Scalar + Div<<A as Scalar>::Real, Output = A> + MulAssign<<A as Scalar>::Real>,
    A::Real: Real,
    S: DataMut<Elem = A>,
{
    let mut x_norm = blas::nrm2(&x);
    if x_norm == A::zero().re() && alpha.im() == A::zero().re() {
        return (alpha.re(), x, A::zero());
    }

    let mut beta = -lapack::lapy3(alpha.re(), alpha.im(), x_norm).copysign(alpha.re());
    let safe_min = A::Real::sfmin() / A::Real::eps();
    let mut knt = 0;
    if beta.abs() < safe_min {
        let safe_min_recip = safe_min.recip();
        loop {
            knt += 1;
            blas::scal(safe_min_recip, &mut x);
            beta *= safe_min_recip;
            alpha *= safe_min_recip;
            if beta.abs() >= safe_min || knt >= 20 {
                break;
            }
        }
        x_norm = blas::nrm2(&x);
        beta = -(alpha.norm_sqr() + x_norm * x_norm).copysign(alpha.re());
    }
    let tau = (beta.into() - alpha) / beta;
    alpha = A::one() / (alpha - beta.into());
    blas::scal(alpha, &mut x);

    beta *= safe_min.powi(knt);
    (beta, x, tau)
}

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq, AbsDiffEq};
    use ndarray::arr1;
    use num_complex::Complex64;

    #[test]
    fn larfg_empty() {
        let x = arr1(&[]);
        let (beta, x, tau) = super::larfg(2., x);
        assert_eq!(beta, 2.);
        assert_eq!(x.len(), 0);
        assert_eq!(tau, 0.);
    }

    #[test]
    fn larfg_real() {
        let x = arr1(&[2., 3., 2.]);
        let (beta, x, tau) = super::larfg(1., x);
        assert_abs_diff_eq!(beta, -4.24264069, epsilon = 1e-6);
        assert!(x.abs_diff_eq(&arr1(&[0.38148714, 0.57223071, 0.38148714]), 1e-6));
        assert_abs_diff_eq!(tau, 1.23570226, epsilon = 1e-6);
    }

    #[test]
    fn larfg_complex() {
        let x = arr1(&[Complex64::new(2., 3.), Complex64::new(3., 2.)]);
        let (beta, x, tau) = super::larfg(Complex64::new(1., 1.), x);
        assert_abs_diff_eq!(beta, -5.2915026221291805, epsilon = 1e-8);
        assert_abs_diff_eq!(x[0].re, 0.38397859, epsilon = 1e-6);
        assert_abs_diff_eq!(x[0].im, 0.41580232, epsilon = 1e-6);
        assert_abs_diff_eq!(x[1].re, 0.51436575, epsilon = 1e-6);
        assert_abs_diff_eq!(x[1].im, 0.23613346, epsilon = 1e-6);
        assert_abs_diff_eq!(tau.re, 1.1889822365046137, epsilon = 1e-8);
        assert_abs_diff_eq!(tau.im, 0.18898223650461363, epsilon = 1e-8);
    }
}
