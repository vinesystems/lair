use crate::{blas, lapack, Real, Scalar};
use ndarray::{s, ArrayViewMut2, Axis};
use std::cmp;
use std::ptr;

#[derive(Debug, PartialEq)]
pub(crate) struct Singular(usize);

pub fn getrf<A>(a: ArrayViewMut2<A>) -> (Vec<usize>, Option<usize>)
where
    A: Scalar,
    A::Real: Real,
{
    unsafe {
        let dim_min = cmp::min(a.nrows(), a.ncols());
        let mut pivots = (0..dim_min).collect::<Vec<_>>();
        let singular = if a.is_standard_layout() {
            getrf_row_major(a, &mut pivots)
        } else {
            getrf_col_major(a, &mut pivots)
        };
        (pivots, singular)
    }
}

#[allow(dead_code)]
pub(crate) fn getrf_recursive<A>(a: ArrayViewMut2<A>) -> Result<Vec<usize>, Singular>
where
    A: Scalar,
    A::Real: Real,
{
    let mut pivots = vec![0; a.nrows()];
    unsafe {
        recursive_inner(a, &mut pivots)?;
    }
    Ok(pivots)
}

/// # Safety
///
/// * `pivots.len()` must be greater than or equal to the smaller dimension of `a`.
#[allow(clippy::cast_possible_wrap)] // The number of elements in the matrix does not exceed `isize::MAX`.
unsafe fn getrf_row_major<A: Scalar>(
    mut a: ArrayViewMut2<A>,
    pivots: &mut [usize],
) -> Option<usize> {
    let mut singular_row = None;
    let dim_min = cmp::min(a.nrows(), a.ncols());
    let row_stride = a.stride_of(Axis(0));
    let col_stride = a.stride_of(Axis(1));
    let a_ptr = a.as_mut_ptr();
    for ul in 0..dim_min {
        let (max_idx, max_val) = blas::iamax(
            a.nrows() - ul,
            a_ptr.offset(row_stride * ul as isize + col_stride * ul as isize),
            row_stride,
        );

        if max_idx != 0 {
            let max_row = max_idx + ul;
            *pivots.get_unchecked_mut(ul) = max_row;
            swap_rows(
                a.ncols(),
                a_ptr.offset(ul as isize * row_stride),
                a_ptr.offset(max_row as isize * row_stride),
                col_stride,
            );
        }
        if max_val == A::zero().re() {
            singular_row = Some(max_idx + ul);
        } else {
            // gerc
            let pivot_recip = A::one() / *a.uget((ul, ul));
            let mut row_j = a_ptr.offset(row_stride * ul as isize + col_stride * ul as isize);
            if col_stride == 1 {
                for j in 1..(a.nrows() - ul) as isize {
                    row_j = row_j.offset(row_stride);
                    *row_j *= pivot_recip;
                    let ratio = *row_j;
                    let mut row_i = a_ptr.offset(row_stride * ul as isize + ul as isize);
                    for _ in ul + 1..a.ncols() {
                        row_i = row_i.offset(1);
                        let elem = ratio * *row_i;
                        *row_i.offset(j * row_stride) -= elem;
                    }
                }
            } else if row_stride == 1 {
                for j in 1..(a.nrows() - ul) as isize {
                    row_j = row_j.offset(1);
                    *row_j *= pivot_recip;
                    let ratio = *row_j;
                    let mut row_i = a_ptr.offset(ul as isize + col_stride * ul as isize);
                    for _ in ul + 1..a.ncols() {
                        row_i = row_i.offset(col_stride);
                        let elem = ratio * *row_i;
                        *row_i.offset(j) -= elem;
                    }
                }
            } else {
                for j in 1..(a.nrows() - ul) as isize {
                    row_j = row_j.offset(row_stride);
                    *row_j *= pivot_recip;
                    let ratio = *row_j;
                    let mut row_i =
                        a_ptr.offset(row_stride * ul as isize + col_stride * ul as isize);
                    for _ in ul + 1..a.ncols() {
                        row_i = row_i.offset(col_stride);
                        let elem = ratio * *row_i;
                        *row_i.offset(j * row_stride) -= elem;
                    }
                }
            }
        }
    }

    singular_row
}

/// # Safety
///
/// * `pivots.len()` must be greater than or equal to the smaller dimension of `a`.
/// * Every element of `pivots` must be a valid row index for `a`.
#[allow(clippy::cast_possible_wrap)] // The number of elements in the matrix does not exceed `isize::MAX`.
#[allow(clippy::too_many_lines)]
unsafe fn getrf_col_major<A>(mut a: ArrayViewMut2<A>, pivots: &mut [usize]) -> Option<usize>
where
    A: Scalar,
{
    let mut singular_row = None;
    let row_stride = a.stride_of(Axis(0));
    let col_stride = a.stride_of(Axis(1));
    let nrows = a.nrows();
    let a_ptr = a.as_mut_ptr();
    for ul in 0..a.ncols() {
        let (left, mut right) = a.view_mut().split_at(Axis(1), ul);
        let mut col = right.column_mut(0);
        let n_upper_rows = cmp::min(ul, nrows);
        for i in 0..n_upper_rows {
            let ip = *pivots.get_unchecked(i);
            if ip != i {
                col.uswap(i, ip);
            }
        }

        for i in 1..n_upper_rows {
            let row = left.row(i);
            let sum = blas::dot(&row, &col, i);
            *col.uget_mut(i) -= sum;
        }

        if ul < nrows {
            let (upper_col, mut lower_col) = col.view_mut().split_at(Axis(0), ul);
            blas::gemv::notrans(
                -A::one(),
                &left.slice(s![ul.., ..]),
                &upper_col,
                A::one(),
                &mut lower_col,
            );
            if row_stride == 1 {
                let (max_row, _) = blas::iamax(nrows - ul, col.as_ptr().add(ul), 1);
                let pivot_row = ul + max_row;
                *pivots.get_unchecked_mut(ul) = pivot_row;
                let pivot = *col.as_ptr().add(pivot_row);
                if pivot == A::zero() {
                    singular_row = Some(ul);
                } else {
                    let pivot_recip = A::one() / pivot;
                    if pivot_row != ul {
                        swap_rows(ul + 1, a_ptr.add(ul), a_ptr.add(pivot_row), col_stride);
                    }
                    if ul + 1 < nrows {
                        for row in ul + 1..nrows {
                            *col.as_mut_ptr().add(row) *= pivot_recip;
                        }
                    }
                }
            } else {
                let (max_row, _) = blas::iamax(
                    nrows - ul,
                    col.as_ptr().offset(ul as isize * row_stride),
                    row_stride,
                );
                let pivot_row = ul + max_row;
                *pivots.get_unchecked_mut(ul) = pivot_row;
                let pivot = *col.as_ptr().offset(pivot_row as isize * row_stride);
                if pivot == A::zero() {
                    singular_row = Some(ul);
                } else {
                    let pivot_recip = A::one() / pivot;
                    if pivot_row != ul {
                        swap_rows(
                            ul + 1,
                            a_ptr.offset(ul as isize * row_stride),
                            a_ptr.offset(pivot_row as isize * row_stride),
                            col_stride,
                        );
                    }
                    if ul + 1 < nrows {
                        for row in ul + 1..nrows {
                            *col.as_mut_ptr().offset(row as isize * row_stride) *= pivot_recip;
                        }
                    }
                }
            }
        }
    }

    singular_row
}

#[allow(clippy::cast_possible_wrap)]
unsafe fn recursive_inner<A>(mut a: ArrayViewMut2<A>, pivots: &mut [usize]) -> Result<(), Singular>
where
    A: Scalar,
    A::Real: Real,
{
    if a.nrows() == 0 || a.ncols() == 0 {
        return Ok(());
    }

    if a.nrows() == 1 {
        *pivots.get_unchecked_mut(0) = 0;
        return if *a.uget((0, 0)) == A::zero() {
            Err(Singular(0))
        } else {
            Ok(())
        };
    }

    if a.ncols() == 1 {
        let (max_idx, max_val) = blas::iamax(a.nrows(), a.as_ptr(), a.stride_of(Axis(0)));
        *pivots.get_unchecked_mut(0) = max_idx;
        if max_val == A::zero().re() {
            return Err(Singular(0));
        }

        if max_idx != 0 {
            a.swap((0, 0), (max_idx, 0));
        }

        if max_val >= A::Real::sfmin() {
            blas::scal(
                A::one() / *a.uget((0, 0)),
                &mut a.column_mut(0).slice_mut(s![1..]),
            );
        } else {
            let pivot = *a.uget((0, 0));
            for a_elem in a.column_mut(0).iter_mut().skip(1) {
                *a_elem /= pivot;
            }
        }

        return Ok(());
    }

    let row_stride = a.stride_of(Axis(0));
    let col_stride = a.stride_of(Axis(1));

    let left_cols = cmp::min(a.nrows(), a.ncols()) / 2;
    let right_cols = a.ncols() - left_cols;
    let mut singular_row = match recursive_inner(a.slice_mut(s![.., ..left_cols]), pivots) {
        Ok(()) => 0,
        Err(Singular(row)) => row,
    };

    lapack::laswp(
        right_cols,
        a.as_mut_ptr().offset(col_stride * left_cols as isize),
        row_stride,
        col_stride,
        0,
        &pivots[0..left_cols],
    );
    blas::trsm(
        a.as_ptr(),
        row_stride,
        col_stride,
        a.slice_mut(s![..left_cols, left_cols..]),
    );
    let a_ptr = a.as_mut_ptr();
    let min_dim = cmp::min(a.nrows(), a.ncols());
    let (upper, lower) = a.split_at(Axis(0), left_cols);
    let upper_right = upper.slice(s![.., left_cols..]);
    let (lower_left, mut lower_right) = lower.split_at(Axis(1), left_cols);
    blas::gemm(
        -A::one(),
        &lower_left,
        false,
        &upper_right,
        false,
        &mut lower_right,
    );
    match recursive_inner(lower_right, &mut pivots[left_cols..]) {
        Ok(()) => {}
        Err(Singular(row)) => {
            if singular_row == 0 {
                singular_row = left_cols + row;
            }
        }
    }
    for p in pivots[left_cols..min_dim].iter_mut() {
        *p += left_cols;
    }
    lapack::laswp(
        left_cols,
        a_ptr,
        row_stride,
        col_stride,
        left_cols,
        &pivots[..min_dim],
    );

    if singular_row == 0 {
        Ok(())
    } else {
        Err(Singular(singular_row))
    }
}

unsafe fn swap_rows<A>(mut n: usize, mut row1: *mut A, mut row2: *mut A, stride: isize) {
    while n > 0 {
        let mut tmp = ptr::read(row1);
        tmp = ptr::replace(row2, tmp);
        ptr::write(row1, tmp);
        row1 = row1.offset(stride);
        row2 = row2.offset(stride);
        n -= 1;
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{arr2, Array2, ArrayBase, Axis};
    use num_complex::Complex64;

    #[test]
    fn singular() {
        let mut a = arr2(&[[1_f64, 1_f64], [1_f64, 1_f64]]);
        let (_, singular) = super::getrf(a.view_mut());
        assert_eq!(singular, Some(1));
    }

    #[test]
    fn empty() {
        let mut a: Array2<f32> = ArrayBase::eye(0);
        let (pivots, singular) = super::getrf(a.view_mut());
        assert_eq!(pivots, Vec::<usize>::new());
        assert!(singular.is_none());
    }

    #[test]
    fn recursive_empty() {
        let mut a: Array2<f32> = ArrayBase::eye(0);
        let p = super::getrf_recursive(a.view_mut()).expect("valid input");
        assert_eq!(p, Vec::<usize>::new());
    }

    #[test]
    fn smallest() {
        let mut a = arr2(&[[3_f64]]);
        let (pivots, singular) = super::getrf(a.view_mut());
        assert_eq!(pivots, vec![0]);
        assert!(singular.is_none());
        assert_eq!(a, arr2(&[[3.]]))
    }

    #[test]
    fn recursive_smallest() {
        let mut a = arr2(&[[3_f64]]);
        let p = super::getrf_recursive(a.view_mut()).expect("valid input");
        assert_eq!(p, vec![0]);
        assert_eq!(a, arr2(&[[3.]]))
    }

    #[test]
    fn recursive_square() {
        let mut a = arr2(&[
            [1_f64, 2_f64, 3_f64, 1_f64, 2_f64],
            [2_f64, 2_f64, 1_f64, 3_f64, 3_f64],
            [3_f64, 1_f64, 2_f64, 2_f64, 1_f64],
            [2_f64, 3_f64, 3_f64, 1_f64, 1_f64],
            [1_f64, 3_f64, 1_f64, 3_f64, 1_f64],
        ]);
        let p = super::getrf_recursive(a.view_mut()).expect("valid input");
        assert_eq!(p, vec![2, 4, 2, 3, 4]);
        assert_abs_diff_eq!(
            a,
            arr2(&[
                [3., 1., 2., 2., 1.],
                [0.33333333, 2.66666667, 0.33333333, 2.33333333, 0.66666667],
                [0.33333333, 0.625, 2.125, -1.125, 1.25],
                [0.66666667, 0.875, 0.64705882, -1.64705882, -1.05882353],
                [0.66666667, 0.5, -0.23529412, -0.14285714, 2.14285714],
            ]),
            epsilon = 1e-6
        );
    }

    #[test]
    fn square_row_major() {
        let mut a = arr2(&[
            [1_f64, 2_f64, 3_f64, 1_f64, 2_f64],
            [2_f64, 2_f64, 1_f64, 3_f64, 3_f64],
            [3_f64, 1_f64, 2_f64, 2_f64, 1_f64],
            [2_f64, 3_f64, 3_f64, 1_f64, 1_f64],
            [1_f64, 3_f64, 1_f64, 3_f64, 1_f64],
        ]);
        let (pivots, singular) = super::getrf(a.view_mut());
        assert_eq!(pivots, vec![2, 4, 2, 3, 4]);
        assert!(singular.is_none());
        assert_abs_diff_eq!(
            a,
            arr2(&[
                [3., 1., 2., 2., 1.],
                [0.33333333, 2.66666667, 0.33333333, 2.33333333, 0.66666667],
                [0.33333333, 0.625, 2.125, -1.125, 1.25],
                [0.66666667, 0.875, 0.64705882, -1.64705882, -1.05882353],
                [0.66666667, 0.5, -0.23529412, -0.14285714, 2.14285714],
            ]),
            epsilon = 1e-6
        );
    }

    #[test]
    fn square_col_major() {
        let mut a = arr2(&[
            [1_f64, 2_f64, 3_f64, 2_f64, 1_f64],
            [2_f64, 2_f64, 1_f64, 3_f64, 3_f64],
            [3_f64, 1_f64, 2_f64, 3_f64, 1_f64],
            [1_f64, 3_f64, 2_f64, 1_f64, 3_f64],
            [2_f64, 3_f64, 1_f64, 1_f64, 1_f64],
        ]);
        a.swap_axes(0, 1);
        assert!(!a.is_standard_layout());
        let (pivots, singular) = super::getrf(a.view_mut());
        assert_eq!(pivots, vec![2, 4, 2, 3, 4]);
        assert!(singular.is_none());
        assert_abs_diff_eq!(
            a,
            arr2(&[
                [3., 1., 2., 2., 1.],
                [0.33333333, 2.66666667, 0.33333333, 2.33333333, 0.66666667],
                [0.33333333, 0.625, 2.125, -1.125, 1.25],
                [0.66666667, 0.875, 0.64705882, -1.64705882, -1.05882353],
                [0.66666667, 0.5, -0.23529412, -0.14285714, 2.14285714],
            ]),
            epsilon = 1e-6
        );
    }

    #[test]
    fn square_negative_strides() {
        let mut a = arr2(&[
            [1_f64, 3_f64, 1_f64, 3_f64, 1_f64],
            [1_f64, 1_f64, 3_f64, 3_f64, 2_f64],
            [1_f64, 2_f64, 2_f64, 1_f64, 3_f64],
            [3_f64, 3_f64, 1_f64, 2_f64, 2_f64],
            [2_f64, 1_f64, 3_f64, 2_f64, 1_f64],
        ]);
        a.invert_axis(Axis(0));
        a.invert_axis(Axis(1));
        assert!(!a.is_standard_layout());
        let (pivots, singular) = super::getrf(a.view_mut());
        assert_eq!(pivots, vec![2, 4, 2, 3, 4]);
        assert!(singular.is_none());
        assert_abs_diff_eq!(
            a,
            arr2(&[
                [3., 1., 2., 2., 1.],
                [0.33333333, 2.66666667, 0.33333333, 2.33333333, 0.66666667],
                [0.33333333, 0.625, 2.125, -1.125, 1.25],
                [0.66666667, 0.875, 0.64705882, -1.64705882, -1.05882353],
                [0.66666667, 0.5, -0.23529412, -0.14285714, 2.14285714],
            ]),
            epsilon = 1e-6
        );
    }

    #[test]
    fn square_complex() {
        let mut a = arr2(&[
            [Complex64::new(1., 1.), Complex64::new(2., -1.)],
            [Complex64::new(3., 1.), Complex64::new(4., -1.)],
        ]);
        let (pivots, singular) = super::getrf(a.view_mut());
        assert_eq!(pivots, &[1, 1]);
        assert!(singular.is_none());
        assert_abs_diff_eq!(a[(0, 0)].re, 3., epsilon = 1e-6);
        assert_abs_diff_eq!(a[(0, 0)].im, 1., epsilon = 1e-6);
        assert_abs_diff_eq!(a[(0, 1)].re, 4., epsilon = 1e-6);
        assert_abs_diff_eq!(a[(0, 1)].im, -1., epsilon = 1e-6);
        assert_abs_diff_eq!(a[(1, 0)].re, 0.4, epsilon = 1e-6);
        assert_abs_diff_eq!(a[(1, 0)].im, 0.2, epsilon = 1e-6);
        assert_abs_diff_eq!(a[(1, 1)].re, 0.2, epsilon = 1e-6);
        assert_abs_diff_eq!(a[(1, 1)].im, -1.4, epsilon = 1e-6);
    }

    #[test]
    fn wide() {
        let mut a = arr2(&[[1_f64, 2_f64, 3_f64], [2_f64, 3_f64, 4_f64]]);
        let (pivots, singular) = super::getrf(a.view_mut());
        assert_eq!(pivots, vec![1, 1]);
        assert!(singular.is_none());
        assert_eq!(a, arr2(&[[2., 3., 4.], [0.5, 0.5, 1.]]));
    }

    #[test]
    fn tall() {
        let mut a = arr2(&[[1_f64, 2_f64], [1_f64, 3_f64], [2_f64, 3_f64]]);
        let (pivots, singular) = super::getrf(a.view_mut());
        assert_eq!(pivots, vec![2, 1]);
        assert!(singular.is_none());
        assert_abs_diff_eq!(
            a,
            arr2(&[[2., 3.], [0.5, 1.5], [0.5, 0.3333333333333333]]),
            epsilon = 1e-6
        );
    }
}
