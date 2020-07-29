use num_traits::Zero;
use std::iter::Sum;
use std::ops::{AddAssign, Mul};

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
    T: Copy + AddAssign + Mul<Output = T> + Sum<<T as Mul>::Output> + Zero,
{
    debug_assert!(n <= isize::MAX as usize);
    debug_assert!((n as isize - 1).checked_mul(inc_x).is_some());
    debug_assert!((n as isize - 1).checked_mul(inc_y).is_some());

    if inc_x == 1 && inc_y == 1 {
        contiguous(n, x, y)
    } else {
        let mut x = x;
        let mut y = y;
        let mut remaining = n;
        let mut sum = if remaining >= 4 {
            let mut sum_0 = T::zero();
            let mut sum_1 = T::zero();
            let mut sum_2 = T::zero();
            let mut sum_3 = T::zero();
            loop {
                sum_0 += *x * *y;
                sum_1 += *x.offset(inc_x) * *y.offset(inc_y);
                sum_2 += *x.offset(inc_x * 2) * *y.offset(inc_y * 2);
                sum_3 += *x.offset(inc_x * 3) * *y.offset(inc_y * 3);
                x = x.offset(inc_x * 4);
                y = y.offset(inc_y * 4);
                remaining -= 4;
                if remaining < 4 {
                    break;
                }
            }
            sum_0 + sum_1 + sum_2 + sum_3
        } else {
            T::zero()
        };
        if remaining > 0 {
            sum += *x * *y;
            x = x.offset(inc_x);
            y = y.offset(inc_y);
        }
        if remaining > 1 {
            sum += *x * *y;
            x = x.offset(inc_x);
            y = y.offset(inc_y);
        }
        if remaining > 2 {
            sum += *x * *y;
        }
        sum
    }
}

unsafe fn contiguous<T>(n: usize, x: *const T, y: *const T) -> T
where
    T: Copy + AddAssign + Mul<Output = T> + Zero,
{
    let mut x = x;
    let mut y = y;
    let mut remaining = n;
    let mut sum = if remaining >= 4 {
        let mut sum_0 = T::zero();
        let mut sum_1 = T::zero();
        let mut sum_2 = T::zero();
        let mut sum_3 = T::zero();
        loop {
            sum_0 += *x * *y;
            sum_1 += *x.add(1) * *y.add(1);
            sum_2 += *x.add(2) * *y.add(2);
            sum_3 += *x.add(3) * *y.add(3);
            x = x.add(4);
            y = y.add(4);
            remaining -= 4;
            if remaining < 4 {
                break;
            }
        }
        sum_0 + sum_1 + sum_2 + sum_3
    } else {
        T::zero()
    };
    if remaining > 0 {
        sum += *x * *y;
        x = x.add(1);
        y = y.add(1);
    }
    if remaining > 1 {
        sum += *x * *y;
        x = x.add(1);
        y = y.add(1);
    }
    if remaining > 2 {
        sum += *x * *y;
    }
    sum
}
