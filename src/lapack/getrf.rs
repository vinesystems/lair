use crate::{blas, lapack, Real, Scalar};
use ndarray::{s, ArrayViewMut2, Axis};
use std::cmp;
use std::ptr;

#[derive(Debug)]
pub(crate) struct Singular(usize);

pub(crate) fn getrf<A>(a: ArrayViewMut2<A>) -> Result<Vec<usize>, Singular>
where
    A: Scalar,
    A::Real: Real,
{
    let row_stride = a.stride_of(Axis(0));
    let col_stride = a.stride_of(Axis(1));
    if col_stride == 1 {
        unsafe { getrf_row_major(a, row_stride) }
    } else if row_stride == 1 {
        unsafe { getrf_col_major(a, col_stride) }
    } else {
        unsafe { getrf_general(a, row_stride, col_stride) }
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

#[allow(clippy::cast_possible_wrap)] // The number of elements in the matrix does not exceed `isize::MAX`.
unsafe fn getrf_row_major<A: Scalar>(
    mut a: ArrayViewMut2<A>,
    lda: isize,
) -> Result<Vec<usize>, Singular> {
    let mut p = (0..a.nrows()).collect::<Vec<_>>();
    let a_ptr = a.as_mut_ptr();
    for i in 0..cmp::min(a.nrows(), a.ncols()) {
        let (max_idx, max_val) = blas::iamax(&a.slice(s![i.., i]));
        if max_val == A::zero().re() {
            return Err(Singular(max_idx));
        }

        if max_idx != 0 {
            let max_row = max_idx + i;
            p.swap(i, max_row);
            ptr::swap_nonoverlapping(
                a_ptr.offset(i as isize * lda),
                a_ptr.offset(max_row as isize * lda),
                a.ncols(),
            );
        }

        let inv_pivot = A::one() / *a.uget((i, i));
        let mut row_j = a_ptr.offset(lda * i as isize + i as isize);
        for j in 1..(a.nrows() - i) as isize {
            row_j = row_j.offset(lda);
            *row_j *= inv_pivot;
            let ratio = *row_j;
            let mut row_i = a_ptr.offset(lda * i as isize + i as isize);
            for _ in i + 1..a.ncols() {
                row_i = row_i.offset(1);
                let elem = ratio * *row_i;
                *row_i.offset(j * lda) -= elem;
            }
        }
    }
    Ok(p)
}

#[allow(clippy::cast_possible_wrap)] // The number of elements in the matrix does not exceed `isize::MAX`.
unsafe fn getrf_col_major<A: Scalar>(
    mut a: ArrayViewMut2<A>,
    lda: isize,
) -> Result<Vec<usize>, Singular> {
    let mut p = (0..a.nrows()).collect::<Vec<_>>();
    let a_ptr = a.as_mut_ptr();
    for i in 0..cmp::min(a.nrows(), a.ncols()) {
        let (max_idx, max_val) = blas::iamax(&a.slice(s![i.., i]));
        if max_val == A::zero().re() {
            return Err(Singular(max_idx));
        }

        if max_idx != 0 {
            let max_row = max_idx + i;
            p.swap(i, max_row);
            swap_rows(a.ncols(), a_ptr.add(i), a_ptr.add(max_row), lda);
        }

        let inv_pivot = A::one() / *a.uget((i, i));
        let mut row_j = a_ptr.offset(i as isize + lda * (i as isize));
        for j in 1..(a.nrows() - i) as isize {
            row_j = row_j.offset(1);
            *row_j *= inv_pivot;
            let ratio = *row_j;
            let mut row_i = a_ptr.offset(i as isize + lda * (i as isize));
            for _ in i + 1..a.ncols() {
                row_i = row_i.offset(lda);
                let elem = ratio * *row_i;
                *row_i.offset(j) -= elem;
            }
        }
    }
    Ok(p)
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
        let (max_idx, max_val) = blas::iamax(&a.slice(s![.., 0]));
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
        Ok(_) => 0,
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
    blas::gemm(
        a.nrows() - left_cols,
        right_cols,
        left_cols,
        a.as_ptr().offset(left_cols as isize * row_stride),
        a.as_ptr().offset(left_cols as isize * col_stride),
        a.as_mut_ptr()
            .offset(left_cols as isize * row_stride + left_cols as isize * col_stride),
        row_stride,
        col_stride,
    );
    match recursive_inner(
        a.slice_mut(s![left_cols.., left_cols..]),
        &mut pivots[left_cols..],
    ) {
        Ok(_) => {}
        Err(Singular(row)) => {
            if singular_row == 0 {
                singular_row = left_cols + row;
            }
        }
    }
    for p in pivots[left_cols..cmp::min(a.nrows(), a.ncols())].iter_mut() {
        *p += left_cols;
    }
    lapack::laswp(
        left_cols,
        a.as_mut_ptr(),
        row_stride,
        col_stride,
        left_cols,
        &pivots[..cmp::min(a.nrows(), a.ncols())],
    );

    if singular_row == 0 {
        Ok(())
    } else {
        Err(Singular(singular_row))
    }
}

#[allow(clippy::cast_possible_wrap)] // The number of elements in the matrix does not exceed `isize::MAX`.
unsafe fn getrf_general<A: Scalar>(
    mut a: ArrayViewMut2<A>,
    row_stride: isize,
    col_stride: isize,
) -> Result<Vec<usize>, Singular> {
    let mut p = (0..a.nrows()).collect::<Vec<_>>();
    let a_ptr = a.as_mut_ptr();
    for i in 0..cmp::min(a.nrows(), a.ncols()) {
        let (max_idx, max_val) = blas::iamax(&a.slice(s![i.., i]));
        if max_val == A::zero().re() {
            return Err(Singular(max_idx));
        }

        if max_idx != 0 {
            let max_row = max_idx + i;
            p.swap(i, max_row);
            swap_rows(
                a.ncols(),
                a_ptr.offset(i as isize * row_stride),
                a_ptr.offset(max_row as isize * row_stride),
                col_stride,
            );
        }

        let inv_pivot = A::one() / *a.uget((i, i));
        let mut row_j = a_ptr.offset(row_stride * i as isize + col_stride * (i as isize));
        for j in 1..(a.nrows() - i) as isize {
            row_j = row_j.offset(row_stride);
            *row_j *= inv_pivot;
            let ratio = *row_j;
            let mut row_i = a_ptr.offset(row_stride * i as isize + col_stride * (i as isize));
            for _ in i + 1..a.ncols() {
                row_i = row_i.offset(col_stride);
                let elem = ratio * *row_i;
                *row_i.offset(j * row_stride) -= elem;
            }
        }
    }
    Ok(p)
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
mod test {
    use approx::{assert_relative_eq, relative_eq, AbsDiffEq};
    use ndarray::{arr2, Array2, ArrayBase, Axis};

    #[test]
    fn singular() {
        let mut a = arr2(&[[1_f64, 1_f64], [1_f64, 1_f64]]);
        assert!(super::getrf(a.view_mut()).is_err());
    }

    #[test]
    fn empty() {
        let mut a: Array2<f32> = ArrayBase::eye(0);
        let p = super::getrf(a.view_mut()).expect("valid input");
        assert_eq!(p, Vec::<usize>::new());
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
        let p = super::getrf(a.view_mut()).expect("valid input");
        assert_eq!(p, vec![0]);
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
        assert!(a.abs_diff_eq(
            &arr2(&[
                [3., 1., 2., 2., 1.],
                [0.33333333, 2.66666667, 0.33333333, 2.33333333, 0.66666667],
                [0.33333333, 0.625, 2.125, -1.125, 1.25],
                [0.66666667, 0.875, 0.64705882, -1.64705882, -1.05882353],
                [0.66666667, 0.5, -0.23529412, -0.14285714, 2.14285714],
            ]),
            1e-6
        ));
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
        let p = super::getrf(a.view_mut()).expect("valid input");
        assert_eq!(p, vec![2, 4, 0, 3, 1]);
        assert_relative_eq!(a[(0, 0)], 3.);
        assert_relative_eq!(a[(1, 0)], 0.3333333333333333);
        assert_relative_eq!(a[(2, 0)], 0.3333333333333333);
        assert_relative_eq!(a[(3, 0)], 0.6666666666666666);
        assert_relative_eq!(a[(4, 0)], 0.6666666666666666);
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
        let p = super::getrf(a.view_mut()).expect("valid input");
        assert_eq!(p, vec![2, 4, 0, 3, 1]);
        assert_relative_eq!(a[(0, 0)], 3.);
        assert_relative_eq!(a[(1, 0)], 0.3333333333333333);
        assert_relative_eq!(a[(2, 0)], 0.3333333333333333);
        assert_relative_eq!(a[(3, 0)], 0.6666666666666666);
        assert_relative_eq!(a[(4, 0)], 0.6666666666666666);
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
        let p = super::getrf(a.view_mut()).expect("valid input");
        assert_eq!(p, vec![2, 4, 0, 3, 1]);
        assert_relative_eq!(a[(0, 0)], 3.);
        assert_relative_eq!(a[(1, 0)], 0.3333333333333333);
        assert_relative_eq!(a[(2, 0)], 0.3333333333333333);
        assert_relative_eq!(a[(3, 0)], 0.6666666666666666);
        assert_relative_eq!(a[(4, 0)], 0.6666666666666666);
    }

    #[test]
    fn wide() {
        let mut a = arr2(&[[1_f64, 2_f64, 3_f64], [2_f64, 3_f64, 4_f64]]);
        let p = super::getrf(a.view_mut()).expect("valid input");
        assert_eq!(p, vec![1, 0]);
        assert_eq!(a, arr2(&[[2., 3., 4.], [0.5, 0.5, 1.]]));
    }

    #[test]
    fn tall() {
        let mut a = arr2(&[[1_f64, 2_f64], [1_f64, 3_f64], [2_f64, 3_f64]]);
        let p = super::getrf(a.view_mut()).expect("valid input");
        assert_eq!(p, vec![2, 1, 0]);
        let lu = arr2(&[[2., 3.], [0.5, 1.5], [0.5, 0.3333333333333333]]);
        assert!(a
            .iter()
            .zip(lu.iter())
            .all(|(a_elem, lu_elem)| relative_eq!(*a_elem, *lu_elem)));
    }
}
