use std::ops::{Mul, MulAssign};

use ndarray::{ArrayBase, Axis, DataMut, Ix1};

use crate::Scalar;

/// Scales a vector by a constant using ndarray iteration.
///
/// Computes `x = a * x` for each element.
pub fn scal<TA, TX, S>(a: TA, x: &mut ArrayBase<S, Ix1>)
where
    TA: Copy,
    TX: MulAssign<TA>,
    S: DataMut<Elem = TX>,
{
    for elem in x.iter_mut() {
        *elem *= a;
    }
}

/// Scales a complex vector by a real constant.
///
/// Computes `zx[i] = da * zx[i]` for each element, where `da` is real
/// and `zx` is a complex vector.
///
/// This is the Rust equivalent of the BLAS `zdscal` routine.
///
/// # Panics
///
/// Panics if `zx` has fewer than `n` elements.
#[allow(dead_code)]
#[inline]
pub(crate) fn rscal<A, S>(n: usize, da: A::Real, zx: &mut ArrayBase<S, Ix1>)
where
    A: Scalar + Mul<A::Real, Output = A>,
    S: DataMut<Elem = A>,
{
    if n == 0 {
        return;
    }
    assert!(zx.len() >= n);

    let inc_x = zx.stride_of(Axis(0));
    if inc_x <= 0 {
        return;
    }
    unsafe { inner(n, da, zx.as_mut_ptr(), inc_x) }
}

/// Scales a complex vector by a real constant.
///
/// # Safety
///
/// * `zx` is the beginning address of an array of at least `n` elements with
///   stride `inc_x`.
/// * The `n` elements of `zx` must have been initialized.
/// * `inc_x` must be positive.
/// * `(n - 1) * inc_x` is between `isize::MIN` and `isize::MAX`, inclusive.
unsafe fn inner<A>(n: usize, da: A::Real, zx: *mut A, inc_x: isize)
where
    A: Scalar + Mul<A::Real, Output = A>,
{
    if inc_x == 1 {
        // Code for increment equal to 1
        for i in 0..n {
            *zx.add(i) = *zx.add(i) * da;
        }
    } else {
        // Code for increment not equal to 1
        let mut ix: isize = 0;
        for _ in 0..n {
            *zx.offset(ix) = *zx.offset(ix) * da;
            ix += inc_x;
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::arr1;
    use num_complex::Complex64;

    use super::rscal;

    #[test]
    fn complex_scale_by_real() {
        let mut zx = arr1(&[
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0),
        ]);

        rscal::<Complex64, _>(3, 2.0, &mut zx);

        assert_eq!(zx[0], Complex64::new(2.0, 4.0));
        assert_eq!(zx[1], Complex64::new(6.0, 8.0));
        assert_eq!(zx[2], Complex64::new(10.0, 12.0));
    }

    #[test]
    fn complex_scale_by_zero() {
        let mut zx = arr1(&[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]);

        rscal::<Complex64, _>(2, 0.0, &mut zx);

        assert_eq!(zx[0], Complex64::new(0.0, 0.0));
        assert_eq!(zx[1], Complex64::new(0.0, 0.0));
    }

    #[test]
    fn complex_scale_by_negative() {
        let mut zx = arr1(&[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]);

        rscal::<Complex64, _>(2, -1.0, &mut zx);

        assert_eq!(zx[0], Complex64::new(-1.0, -2.0));
        assert_eq!(zx[1], Complex64::new(-3.0, -4.0));
    }

    #[test]
    fn complex_zero_elements() {
        let mut zx = arr1(&[Complex64::new(1.0, 2.0)]);

        rscal::<Complex64, _>(0, 5.0, &mut zx);

        // Array should be unchanged
        assert_eq!(zx[0], Complex64::new(1.0, 2.0));
    }

    #[test]
    fn complex_partial_scale() {
        let mut zx = arr1(&[
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0),
        ]);

        // Only scale first 2 elements
        rscal::<Complex64, _>(2, 2.0, &mut zx);

        assert_eq!(zx[0], Complex64::new(2.0, 4.0));
        assert_eq!(zx[1], Complex64::new(6.0, 8.0));
        assert_eq!(zx[2], Complex64::new(5.0, 6.0)); // Unchanged
    }

    #[test]
    fn real_scale() {
        let mut x = arr1(&[1.0, 2.0, 3.0]);

        rscal::<f64, _>(3, 2.5, &mut x);

        assert_abs_diff_eq!(x[0], 2.5, epsilon = 1e-10);
        assert_abs_diff_eq!(x[1], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[2], 7.5, epsilon = 1e-10);
    }

    #[test]
    fn complex_scale_by_fractional() {
        let mut zx = arr1(&[Complex64::new(4.0, 8.0), Complex64::new(6.0, 10.0)]);

        rscal::<Complex64, _>(2, 0.5, &mut zx);

        assert_eq!(zx[0], Complex64::new(2.0, 4.0));
        assert_eq!(zx[1], Complex64::new(3.0, 5.0));
    }
}
