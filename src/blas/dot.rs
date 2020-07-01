use std::iter::Sum;
use std::ops::Mul;

/// Computes a dot product of two vectors.
///
/// # Safety
///
/// * `x` is the beginning address of an array of at least `n` elements with
///   stride `inc_x`.
/// * `y` is the beginning address of an array of at least `n` elements with
///   stride `inc_y`.
/// * The `n` elements of `x` and `y` must have been initialized.
/// * `(n - 1) * inc_x` and `(n - 1) * inc_y` are between `isize::MIN` and
///   `isize::MAX`, inclusive.
#[allow(clippy::cast_possible_wrap)]
pub unsafe fn dot<T>(n: usize, x: *const T, inc_x: isize, y: *const T, inc_y: isize) -> T
where
    T: Copy + Mul + Sum<<T as Mul>::Output>,
{
    debug_assert!(n <= isize::MAX as usize);
    debug_assert!((n as isize - 1).checked_mul(inc_x).is_some());
    debug_assert!((n as isize - 1).checked_mul(inc_y).is_some());

    (0..n)
        .map(|i| *x.offset(inc_x * i as isize) * *y.offset(inc_y * i as isize))
        .sum()
}
