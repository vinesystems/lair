use crate::{blas, lapack, Real, Scalar};
use ndarray::{ArrayBase, DataMut, Ix1};

/// Generates an elementary reflector (Householder matrix).
pub fn larfg<A, S>(mut alpha: A, mut x: ArrayBase<S, Ix1>) -> (A::Real, ArrayBase<S, Ix1>, A)
where
    A: Scalar,
    A::Real: Real,
    S: DataMut<Elem = A>,
{
    if x.is_empty() {
        return (alpha.re(), x, A::zero());
    }

    let mut x_norm = blas::nrm2(&x);
    if x_norm == A::zero().re() {
        return (alpha.re(), x, A::zero());
    }

    let mut beta = -lapack::lapy3(alpha.re(), alpha.im(), x_norm).copysign(alpha.re());
    let safe_min = A::Real::sfmin() / A::Real::eps();
    let safe_min_recip = safe_min.recip();
    let mut knt = 0;
    if beta.abs() < safe_min {
        loop {
            knt += 1;
            blas::rscal(safe_min_recip, &mut x);
            beta *= safe_min_recip;
            alpha = alpha.mul_real(safe_min_recip);
            if beta.abs() >= safe_min || knt >= 20 {
                break;
            }
        }
        x_norm = blas::nrm2(&x);
        beta = -(alpha.square() + x_norm.square()).copysign(alpha.re());
    }
    let tau = (A::from_real(beta) - alpha).div_real(beta);
    alpha = A::one() / (alpha - A::from_real(beta));
    blas::scal(alpha, &mut x);

    beta *= safe_min.powi(knt);
    (beta, x, tau)
}

#[cfg(test)]
mod test {
    use approx::{assert_abs_diff_eq, AbsDiffEq};
    use ndarray::arr1;

    #[test]
    fn larfg_empty() {
        let x = arr1(&[]);
        let (beta, x, tau) = super::larfg(2., x);
        assert_eq!(beta, 2.);
        assert_eq!(x.len(), 0);
        assert_eq!(tau, 0.);
    }

    #[test]
    fn larfg() {
        let x = arr1(&[2., 3., 2.]);
        let (beta, x, tau) = super::larfg(1., x);
        assert_abs_diff_eq!(beta, -4.24264069, epsilon = 1e-6);
        assert!(x.abs_diff_eq(&arr1(&[0.38148714, 0.57223071, 0.38148714]), 1e-6));
        assert_abs_diff_eq!(tau, 1.23570226, epsilon = 1e-6);
    }
}
