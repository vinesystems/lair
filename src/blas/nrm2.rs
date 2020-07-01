use crate::{Float, Scalar};
use ndarray::{ArrayBase, Data, Ix1};

pub(crate) fn nrm2<A, S>(x: &ArrayBase<S, Ix1>) -> A::Real
where
    A: Scalar,
    S: Data<Elem = A>,
{
    x.iter()
        .map(|v| v.re() * v.re() + v.im() * v.im())
        .sum::<A::Real>()
        .sqrt()
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;
    use ndarray::arr1;
    use num_complex::Complex64;

    #[test]
    fn real() {
        let x = arr1(&[3., 4.]);
        let norm = super::nrm2(&x);
        assert_eq!(norm, 5.);
    }

    #[test]
    fn complex() {
        let x = arr1(&[Complex64::new(1., 2.), Complex64::new(3., 4.)]);
        let norm = super::nrm2(&x);
        assert_abs_diff_eq!(norm, 5.477225575051661, epsilon = 1e-8);
    }
}
