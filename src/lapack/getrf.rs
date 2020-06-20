use crate::{blas, Scalar};
use ndarray::{s, ArrayViewMut2};
use std::cmp;
use std::ptr;

#[derive(Debug)]
pub(crate) struct Singular();

#[allow(clippy::cast_possible_wrap)] // The number of elements in the matrix does not exceed `isize::MAX`.
pub(crate) fn getrf<A>(mut a: ArrayViewMut2<A>) -> Result<Vec<usize>, Singular>
where
    A: Scalar,
{
    let mut p = (0..a.nrows()).collect::<Vec<_>>();
    let strides = a.strides();
    let row_stride = strides[0];
    let col_stride = strides[1];
    let a_ptr = a.as_mut_ptr();
    unsafe {
        for i in 0..cmp::min(a.nrows(), a.ncols()) {
            let (max_idx, max_val) = blas::iamax(&a.slice(s![i.., i]));
            if max_val == A::zero().re() {
                return Err(Singular());
            }

            if max_idx == 0 {
                let inv_pivot = A::one() / *a.uget((i, i));
                let mut row_j = a_ptr.offset(row_stride * i as isize + col_stride * (i as isize));
                for j in 1..(a.nrows() - i) as isize {
                    row_j = row_j.offset(row_stride);
                    *row_j *= inv_pivot;
                    let ratio = *row_j;
                    let mut row_i =
                        a_ptr.offset(row_stride * i as isize + col_stride * (i as isize));
                    for _ in i + 1..a.ncols() {
                        row_i = row_i.offset(col_stride);
                        let elem = ratio * *row_i;
                        *row_i.offset(j * row_stride) -= elem;
                    }
                }
            } else {
                let max_row = max_idx + i;
                p.swap(i, max_row);
                swap_rows(
                    a.ncols(),
                    a_ptr.offset(i as isize * row_stride),
                    a_ptr.offset(max_row as isize * row_stride),
                    col_stride,
                );

                let inv_pivot = A::one() / *a.uget((i, i));
                let mut row_j = a_ptr.offset(row_stride * i as isize + col_stride * (i as isize));
                for j in 1..(a.nrows() - i) as isize {
                    row_j = row_j.offset(row_stride);
                    *row_j *= inv_pivot;
                    let ratio = *row_j;
                    let mut row_i =
                        a_ptr.offset(row_stride * i as isize + col_stride * (i as isize));
                    for _ in i + 1..a.ncols() {
                        row_i = row_i.offset(col_stride);
                        let elem = ratio * *row_i;
                        *row_i.offset(j * row_stride) -= elem;
                    }
                }
            }
        }
    }
    Ok(p)
}

unsafe fn swap_rows<A>(mut n: usize, mut row1: *mut A, mut row2: *mut A, stride: isize) {
    if stride == 1 {
        ptr::swap_nonoverlapping(row1, row2, n);
    } else {
        while n > 0 {
            let mut tmp = ptr::read(row1);
            tmp = ptr::replace(row2, tmp);
            ptr::write(row1, tmp);
            row1 = row1.offset(stride);
            row2 = row2.offset(stride);
            n -= 1;
        }
    }
}

#[cfg(test)]
mod test {
    use approx::{assert_relative_eq, relative_eq};
    use ndarray::{arr2, Array2, ArrayBase};

    #[test]
    fn singular() {
        let mut a = arr2(&[[1_f64, 1_f64], [1_f64, 1_f64]]);
        assert!(super::getrf(a.view_mut()).is_err());
    }

    #[test]
    fn empty() {
        let mut a: Array2<f32> = ArrayBase::eye(0);
        let p = super::getrf(a.view_mut()).expect("valid input");
        assert_eq!(p, vec![]);
    }

    #[test]
    fn smallest() {
        let mut a = arr2(&[[3_f64]]);
        let p = super::getrf(a.view_mut()).expect("valid input");
        assert_eq!(p, vec![0]);
        assert_eq!(a, arr2(&[[3.]]))
    }

    #[test]
    fn square() {
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
