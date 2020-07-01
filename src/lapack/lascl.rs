use crate::{Float, One, Real, Scalar, Zero};

/// Multiplies a full matrix by a real scalar `c_to / c_from`.
#[allow(clippy::cast_possible_wrap)]
#[allow(dead_code)]
pub(crate) unsafe fn full<T>(
    c_from: T::Real,
    c_to: T::Real,
    m: usize,
    n: usize,
    a: *mut T,
    row_stride: isize,
    col_stride: isize,
) where
    T: Scalar,
{
    let small_num = T::Real::sfmin();
    let big_num = T::Real::one() / small_num;

    let mut c_from = c_from;
    let mut c_to = c_to;

    loop {
        let (mul, done) = {
            let small_from = c_from * small_num;
            if small_from == c_from {
                ((c_to / c_from).into(), true)
            } else {
                let small_to = c_to / big_num;
                if small_to == c_to {
                    c_from = T::Real::one();
                    ((c_to).into(), true)
                } else if small_from.abs() > c_to.abs() && c_to.abs() != T::Real::zero() {
                    c_from = small_from;
                    ((small_num).into(), false)
                } else if small_to.abs() > c_from {
                    c_to = small_to;
                    ((big_num).into(), false)
                } else {
                    ((c_to / c_from).into(), true)
                }
            }
        };

        for i in 0..m {
            for j in 0..n {
                *a.offset(i as isize * row_stride + j as isize * col_stride) *= mul;
            }
        }

        if done {
            break;
        }
    }
}
