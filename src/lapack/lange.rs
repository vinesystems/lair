use std::cmp::Ordering;

use ndarray::{ArrayBase, Data, Ix2};
use num_traits::{Float, Zero};

use crate::Scalar;

#[allow(dead_code)]
pub fn maxabs<A, S>(a: &ArrayBase<S, Ix2>) -> A::Real
where
    A: Scalar,
    S: Data<Elem = A>,
{
    if a.is_empty() {
        return A::Real::zero();
    }

    a.into_iter()
        .map(Scalar::abs)
        .fold(A::Real::neg_infinity(), |max, v| {
            match max.partial_cmp(&v) {
                None => A::Real::nan(),
                Some(Ordering::Less) => v,
                Some(_) => max,
            }
        })
}

#[cfg(test)]
mod tests {
    use ndarray::{arr2, Array2};
    use num_complex::Complex32;

    #[test]
    fn maxabs_empty() {
        assert_eq!(super::maxabs(&Array2::<f64>::ones((1, 0))), 0.);
    }

    #[test]
    fn maxabs_real() {
        let a = arr2(&[[0., 1.], [-1., -2.]]);
        assert_eq!(super::maxabs(&a), 2.);
    }

    #[test]
    fn maxabs_complex() {
        let a = arr2(&[
            [Complex32::new(0., 2.), Complex32::new(-1., -1.)],
            [Complex32::new(1., 0.), Complex32::new(1., -1.)],
        ]);
        assert_eq!(super::maxabs(&a), 2.);
    }

    #[test]
    fn maxabs_nan() {
        let a = arr2(&[[0., 1.], [-1., f32::NAN]]);
        assert!(super::maxabs(&a).is_nan());
    }
}
