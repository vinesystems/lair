use std::ops::{AddAssign, Mul, Sub};

use ndarray::{ArrayBase, Axis, DataMut, Ix1};

use crate::Scalar;

/// Applies a plane rotation to complex vectors with real cosine and sine.
///
/// For each pair of elements `(cx[i], cy[i])`, computes:
/// - `cx[i] = c * cx[i] + s * cy[i]`
/// - `cy[i] = c * cy[i] - s * cx[i]`
///
/// where `c` (cosine) and `s` (sine) are real, and `cx` and `cy` are complex
/// vectors.
///
/// # Panics
///
/// Panics if `cx` or `cy` has fewer than `n` elements.
#[allow(dead_code)]
#[inline]
pub(crate) fn rot<A, S>(
    n: usize,
    cx: &mut ArrayBase<S, Ix1>,
    cy: &mut ArrayBase<S, Ix1>,
    c: A::Real,
    s: A::Real,
) where
    A: Scalar + AddAssign + Mul<A::Real, Output = A> + Sub<Output = A>,
    S: DataMut<Elem = A>,
{
    assert!(cx.len() >= n && cy.len() >= n);

    let inc_x = cx.stride_of(Axis(0));
    let inc_y = cy.stride_of(Axis(0));
    unsafe { inner(n, cx.as_mut_ptr(), inc_x, cy.as_mut_ptr(), inc_y, c, s) }
}

/// Applies a plane rotation to complex vectors with real cosine and sine.
///
/// # Safety
///
/// * `cx` is the beginning address of an array of at least `n` elements with
///   stride `inc_x`.
/// * `cy` is the beginning address of an array of at least `n` elements with
///   stride `inc_y`.
/// * The `n` elements of `cx` and `cy` must have been initialized.
/// * `(n - 1) * inc_x` and `(n - 1) * inc_y` are between `isize::MIN` and
///   `isize::MAX`, inclusive.
#[allow(clippy::cast_possible_wrap)]
unsafe fn inner<A>(
    n: usize,
    cx: *mut A,
    inc_x: isize,
    cy: *mut A,
    inc_y: isize,
    c: A::Real,
    s: A::Real,
) where
    A: Scalar + AddAssign + Mul<A::Real, Output = A> + Sub<Output = A>,
{
    if n == 0 {
        return;
    }

    if inc_x == 1 && inc_y == 1 {
        // Code for both increments equal to 1
        for i in 0..n {
            let ctemp = *cx.add(i) * c + *cy.add(i) * s;
            *cy.add(i) = *cy.add(i) * c - *cx.add(i) * s;
            *cx.add(i) = ctemp;
        }
    } else {
        // Code for unequal increments or equal increments not equal to 1
        let mut ix: isize = 0;
        let mut iy: isize = 0;
        if inc_x < 0 {
            ix = (-(n as isize) + 1) * inc_x;
        }
        if inc_y < 0 {
            iy = (-(n as isize) + 1) * inc_y;
        }
        for _ in 0..n {
            let ctemp = *cx.offset(ix) * c + *cy.offset(iy) * s;
            *cy.offset(iy) = *cy.offset(iy) * c - *cx.offset(ix) * s;
            *cx.offset(ix) = ctemp;
            ix += inc_x;
            iy += inc_y;
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::arr1;
    use num_complex::Complex64;

    use super::rot;

    #[test]
    fn complex_unit_increment() {
        let mut cx = arr1(&[
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0),
        ]);
        let mut cy = arr1(&[
            Complex64::new(7.0, 8.0),
            Complex64::new(9.0, 10.0),
            Complex64::new(11.0, 12.0),
        ]);

        // Use c = cos(pi/4) = sqrt(2)/2, s = sin(pi/4) = sqrt(2)/2
        let c = std::f64::consts::FRAC_1_SQRT_2;
        let s = std::f64::consts::FRAC_1_SQRT_2;

        rot::<Complex64, _>(3, &mut cx, &mut cy, c, s);

        // After rotation:
        // cx[i] = c * cx_orig[i] + s * cy_orig[i]
        // cy[i] = c * cy_orig[i] - s * cx_orig[i]

        // For i = 0:
        // cx[0] = (1+2i)/sqrt(2) + (7+8i)/sqrt(2) = (8+10i)/sqrt(2)
        // cy[0] = (7+8i)/sqrt(2) - (1+2i)/sqrt(2) = (6+6i)/sqrt(2)
        assert_abs_diff_eq!(cx[0].re, 8.0 * c, epsilon = 1e-10);
        assert_abs_diff_eq!(cx[0].im, 10.0 * c, epsilon = 1e-10);
        assert_abs_diff_eq!(cy[0].re, 6.0 * c, epsilon = 1e-10);
        assert_abs_diff_eq!(cy[0].im, 6.0 * c, epsilon = 1e-10);

        // For i = 1:
        // cx[1] = (3+4i)/sqrt(2) + (9+10i)/sqrt(2) = (12+14i)/sqrt(2)
        // cy[1] = (9+10i)/sqrt(2) - (3+4i)/sqrt(2) = (6+6i)/sqrt(2)
        assert_abs_diff_eq!(cx[1].re, 12.0 * c, epsilon = 1e-10);
        assert_abs_diff_eq!(cx[1].im, 14.0 * c, epsilon = 1e-10);
        assert_abs_diff_eq!(cy[1].re, 6.0 * c, epsilon = 1e-10);
        assert_abs_diff_eq!(cy[1].im, 6.0 * c, epsilon = 1e-10);

        // For i = 2:
        // cx[2] = (5+6i)/sqrt(2) + (11+12i)/sqrt(2) = (16+18i)/sqrt(2)
        // cy[2] = (11+12i)/sqrt(2) - (5+6i)/sqrt(2) = (6+6i)/sqrt(2)
        assert_abs_diff_eq!(cx[2].re, 16.0 * c, epsilon = 1e-10);
        assert_abs_diff_eq!(cx[2].im, 18.0 * c, epsilon = 1e-10);
        assert_abs_diff_eq!(cy[2].re, 6.0 * c, epsilon = 1e-10);
        assert_abs_diff_eq!(cy[2].im, 6.0 * c, epsilon = 1e-10);
    }

    #[test]
    fn complex_zero_elements() {
        let mut cx = arr1(&[Complex64::new(1.0, 2.0)]);
        let mut cy = arr1(&[Complex64::new(3.0, 4.0)]);

        let c = 0.6;
        let s = 0.8;

        rot::<Complex64, _>(0, &mut cx, &mut cy, c, s);

        // Arrays should be unchanged
        assert_eq!(cx[0], Complex64::new(1.0, 2.0));
        assert_eq!(cy[0], Complex64::new(3.0, 4.0));
    }

    #[test]
    fn real_vectors() {
        let mut x = arr1(&[1.0, 2.0, 3.0]);
        let mut y = arr1(&[4.0, 5.0, 6.0]);

        let c = 0.6;
        let s = 0.8;

        rot::<f64, _>(3, &mut x, &mut y, c, s);

        // x[0] = 0.6 * 1 + 0.8 * 4 = 0.6 + 3.2 = 3.8
        // y[0] = 0.6 * 4 - 0.8 * 1 = 2.4 - 0.8 = 1.6
        assert_abs_diff_eq!(x[0], 3.8, epsilon = 1e-10);
        assert_abs_diff_eq!(y[0], 1.6, epsilon = 1e-10);

        // x[1] = 0.6 * 2 + 0.8 * 5 = 1.2 + 4.0 = 5.2
        // y[1] = 0.6 * 5 - 0.8 * 2 = 3.0 - 1.6 = 1.4
        assert_abs_diff_eq!(x[1], 5.2, epsilon = 1e-10);
        assert_abs_diff_eq!(y[1], 1.4, epsilon = 1e-10);

        // x[2] = 0.6 * 3 + 0.8 * 6 = 1.8 + 4.8 = 6.6
        // y[2] = 0.6 * 6 - 0.8 * 3 = 3.6 - 2.4 = 1.2
        assert_abs_diff_eq!(x[2], 6.6, epsilon = 1e-10);
        assert_abs_diff_eq!(y[2], 1.2, epsilon = 1e-10);
    }

    #[test]
    fn identity_rotation() {
        let mut cx = arr1(&[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]);
        let mut cy = arr1(&[Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)]);

        // c = 1, s = 0 means no rotation
        rot::<Complex64, _>(2, &mut cx, &mut cy, 1.0, 0.0);

        assert_eq!(cx[0], Complex64::new(1.0, 2.0));
        assert_eq!(cx[1], Complex64::new(3.0, 4.0));
        assert_eq!(cy[0], Complex64::new(5.0, 6.0));
        assert_eq!(cy[1], Complex64::new(7.0, 8.0));
    }

    #[test]
    fn swap_rotation() {
        let mut cx = arr1(&[Complex64::new(1.0, 2.0)]);
        let mut cy = arr1(&[Complex64::new(3.0, 4.0)]);

        // c = 0, s = 1 swaps (with sign flip on cy)
        rot::<Complex64, _>(1, &mut cx, &mut cy, 0.0, 1.0);

        // cx = 0 * (1+2i) + 1 * (3+4i) = 3+4i
        // cy = 0 * (3+4i) - 1 * (1+2i) = -1-2i
        assert_eq!(cx[0], Complex64::new(3.0, 4.0));
        assert_eq!(cy[0], Complex64::new(-1.0, -2.0));
    }
}
