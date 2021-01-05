use crate::{blas, lapack, Scalar};
use ndarray::{ArrayViewMut, ShapeBuilder};
use std::cmp;
use std::ops::{Div, MulAssign};

/// Reduces the first `width` rows and columns of a matrix to a bidiagonal form.
pub(crate) unsafe fn labrd<T>(
    n_rows: usize,
    n_cols: usize,
    width: usize,
    a: *mut T,
    a_row_stride: isize,
    a_col_stride: isize,
    d: *mut T,
    e: *mut T,
    tau_q: *mut T,
    tau_p: *mut T,
    x: *mut T,
    x_row_stride: isize,
    x_col_stride: isize,
    y: *mut T,
    y_row_stride: isize,
    y_col_stride: isize,
) where
    T: Scalar + Div<<T as Scalar>::Real, Output = T> + MulAssign<<T as Scalar>::Real>,
{
    if n_rows >= n_cols {
        for i in 0..width {
            blas::gemv::notrans(
                n_rows - i,
                i,
                -T::one(),
                a.offset(i as isize * a_row_stride),
                a_row_stride,
                a_col_stride,
                y.offset(i as isize * y_row_stride),
                y_col_stride,
                T::one(),
                a.offset(i as isize * a_row_stride + i as isize * a_col_stride),
                a_row_stride,
            );
            blas::gemv::notrans(
                n_rows - i,
                i,
                -T::one(),
                x.offset(i as isize * x_row_stride),
                x_row_stride,
                x_col_stride,
                a.offset(i as isize * a_col_stride),
                a_row_stride,
                T::one(),
                a.offset(i as isize * a_row_stride + i as isize * a_col_stride),
                a_row_stride,
            );
            let (beta, _, tau) = lapack::larfg(
                *a.offset(i as isize * a_row_stride + i as isize * a_col_stride),
                ArrayViewMut::from_shape_ptr(
                    [n_rows - i].strides([a_row_stride as usize]),
                    a.offset(
                        cmp::min(i + 1, n_rows - 1) as isize * a_row_stride
                            + i as isize * a_col_stride,
                    ),
                ),
            );
            *a.offset(i as isize * a_row_stride + i as isize * a_col_stride) = beta.into();
            *d.add(i) = beta.into();
            *tau_q.offset(i as isize) = tau;
            if i + 1 < n_cols {
                *a.offset(i as isize * a_row_stride + i as isize * a_col_stride) = T::one();

                blas::gemv::conjtrans(
                    n_rows - i,
                    n_cols - i - 1,
                    T::one(),
                    a.offset(i as isize * a_row_stride + (i + 1) as isize * a_col_stride),
                    a_row_stride,
                    a_col_stride,
                    a.offset(i as isize * a_row_stride + i as isize * a_col_stride),
                    a_row_stride,
                    T::zero(),
                    y.offset((i + 1) as isize * y_row_stride + i as isize * y_col_stride),
                    y_row_stride,
                );
                blas::gemv::conjtrans(
                    n_rows - i,
                    i,
                    T::one(),
                    a.offset(i as isize * a_row_stride),
                    a_row_stride,
                    a_col_stride,
                    a.offset(i as isize * a_row_stride + i as isize * a_col_stride),
                    a_row_stride,
                    T::zero(),
                    y.offset(i as isize * y_col_stride),
                    y_row_stride,
                );
                blas::gemv::notrans(
                    n_cols - i - 1,
                    i,
                    -T::one(),
                    y.offset((i + 1) as isize * y_row_stride),
                    y_row_stride,
                    y_col_stride,
                    y.offset(i as isize * y_col_stride),
                    y_row_stride,
                    T::one(),
                    y.offset((i + 1) as isize * y_row_stride + i as isize * y_col_stride),
                    y_row_stride,
                );
                blas::gemv::conjtrans(
                    n_rows - i,
                    i,
                    T::one(),
                    x.offset(i as isize * x_row_stride),
                    x_row_stride,
                    x_col_stride,
                    a.offset(i as isize * a_row_stride + i as isize * a_col_stride),
                    a_row_stride,
                    T::zero(),
                    y.offset(i as isize * y_col_stride),
                    y_row_stride,
                );
                blas::gemv::conjtrans(
                    i,
                    n_cols - i - 1,
                    -T::one(),
                    a.offset((i + 1) as isize * a_col_stride),
                    a_row_stride,
                    a_col_stride,
                    y.offset(i as isize * y_col_stride),
                    y_row_stride,
                    T::one(),
                    y.offset((i + 1) as isize * y_row_stride + i as isize * y_col_stride),
                    y_row_stride,
                );
                blas::scal(
                    *tau_q.add(i),
                    &mut ArrayViewMut::from_shape_ptr(
                        [n_cols - i - 1].strides([y_row_stride as usize]),
                        y.offset((i + 1) as isize * y_row_stride + i as isize * y_col_stride),
                    ),
                );

                blas::gemv::notrans(
                    n_cols - i - 1,
                    i + 1,
                    -T::one(),
                    y.offset((i + 1) as isize * y_row_stride),
                    y_row_stride,
                    y_col_stride,
                    a.offset(i as isize * a_row_stride),
                    a_col_stride,
                    T::one(),
                    a.offset(i as isize * a_row_stride + (i + 1) as isize * a_col_stride),
                    a_col_stride,
                );
                blas::gemv::conjtrans(
                    i,
                    n_cols - i - 1,
                    -T::one(),
                    a.offset((i + 1) as isize * a_row_stride),
                    a_row_stride,
                    a_col_stride,
                    x.offset(i as isize * x_row_stride),
                    x_col_stride,
                    T::one(),
                    a.offset(i as isize * a_row_stride + (i + 1) as isize * a_col_stride),
                    a_col_stride,
                );

                let (beta, _, tau) = lapack::larfg(
                    *a.offset(i as isize * a_row_stride + (i + 1) as isize * a_col_stride),
                    ArrayViewMut::from_shape_ptr(
                        [n_cols - i - 1].strides([a_col_stride as usize]),
                        a.offset(
                            i as isize * a_row_stride
                                + cmp::min(i + 2, n_cols - 1) as isize * a_col_stride,
                        ),
                    ),
                );
                *tau_p.add(i) = tau;
                *e.add(i) = beta.into();
                *a.offset(i as isize * a_row_stride + (i + 1) as isize * a_col_stride) = T::one();

                blas::gemv::notrans(
                    n_rows - i - 1,
                    n_cols - i - 1,
                    T::one(),
                    a.offset((i + 1) as isize * a_row_stride + (i + 1) as isize * a_col_stride),
                    a_row_stride,
                    a_col_stride,
                    a.offset(i as isize * a_row_stride + (i + 1) as isize * a_col_stride),
                    a_col_stride,
                    T::zero(),
                    x.offset((i + 1) as isize * x_row_stride + i as isize * x_col_stride),
                    x_row_stride,
                );
                blas::gemv::conjtrans(
                    n_cols - i - 1,
                    i - 1,
                    T::one(),
                    y.offset((i + 1) as isize * y_row_stride),
                    y_row_stride,
                    y_col_stride,
                    a.offset(i as isize * a_row_stride + (i + 1) as isize * a_col_stride),
                    a_col_stride,
                    T::zero(),
                    x.offset(i as isize * a_col_stride),
                    x_row_stride,
                );
                blas::gemv::notrans(
                    n_rows - i - 1,
                    i + 1,
                    -T::one(),
                    a.offset((i + 1) as isize * a_row_stride + i as isize * a_col_stride),
                    a_row_stride,
                    a_col_stride,
                    x.offset(i as isize * x_col_stride),
                    x_row_stride,
                    T::one(),
                    x.offset((i + 1) as isize * x_row_stride + i as isize * x_col_stride),
                    x_row_stride,
                );
                blas::gemv::notrans(
                    i,
                    n_cols - i + 1,
                    T::one(),
                    a.offset((i + 1) as isize * a_col_stride),
                    a_row_stride,
                    a_col_stride,
                    a.offset(i as isize * a_row_stride + (i + 1) as isize * a_col_stride),
                    a_col_stride,
                    T::zero(),
                    x.offset(i as isize * x_col_stride),
                    x_row_stride,
                );
                blas::gemv::notrans(
                    n_rows - i - 1,
                    i,
                    -T::one(),
                    x.offset((i + 1) as isize * x_row_stride),
                    x_row_stride,
                    x_col_stride,
                    x.offset(i as isize * x_col_stride),
                    x_row_stride,
                    T::one(),
                    x.offset((i + 1) as isize * a_row_stride + i as isize * a_col_stride),
                    x_row_stride,
                );
                blas::scal(
                    *tau_p.add(i),
                    &mut ArrayViewMut::from_shape_ptr(
                        [n_rows - i - 1].strides([x_row_stride as usize]),
                        x.offset((i + 1) as isize * x_row_stride + i as isize * x_col_stride),
                    ),
                );
            }
        }
    } else {
        for i in 0..width {
            blas::gemv::notrans(
                n_cols - i,
                i,
                -T::one(),
                y.offset(i as isize * y_row_stride),
                y_row_stride,
                y_col_stride,
                a.offset(i as isize * a_row_stride),
                a_col_stride,
                T::one(),
                a.offset(i as isize * a_row_stride + i as isize * a_col_stride),
                a_col_stride,
            );
            blas::gemv::conjtrans(
                i,
                n_cols - i,
                -T::one(),
                a.offset(i as isize * a_col_stride),
                a_row_stride,
                a_col_stride,
                x.offset(i as isize * x_row_stride),
                x_col_stride,
                T::one(),
                a.offset(i as isize * a_row_stride + i as isize * a_col_stride),
                a_col_stride,
            );
            let (beta, _, tau) = lapack::larfg(
                *a.offset(i as isize * a_row_stride + i as isize * a_col_stride),
                ArrayViewMut::from_shape_ptr(
                    [n_cols - i].strides([a_col_stride as usize]),
                    a.offset(
                        i as isize * a_row_stride
                            + cmp::min(i + 1, n_cols - 1) as isize * a_col_stride,
                    ),
                ),
            );
            *a.offset(i as isize * a_row_stride + i as isize * a_col_stride) = beta.into();
            *d.add(i) = beta.into();
            *tau_p.offset(i as isize) = tau;
            if i + 1 < n_rows {
                *a.offset(i as isize * a_row_stride + i as isize * a_col_stride) = T::one();

                blas::gemv::notrans(
                    n_rows - i - 1,
                    n_cols - i,
                    T::one(),
                    a.offset((i + 1) as isize * a_row_stride + i as isize * a_col_stride),
                    a_row_stride,
                    a_col_stride,
                    a.offset(i as isize * a_row_stride + i as isize * a_col_stride),
                    a_col_stride,
                    T::zero(),
                    x.offset((i + 1) as isize * x_row_stride + i as isize * x_col_stride),
                    x_row_stride,
                );
                blas::gemv::conjtrans(
                    n_cols - i,
                    i,
                    T::one(),
                    y.offset(i as isize * y_row_stride),
                    y_row_stride,
                    y_col_stride,
                    a.offset(i as isize * a_row_stride + i as isize * a_col_stride),
                    a_col_stride,
                    T::zero(),
                    x.offset(i as isize * x_col_stride),
                    x_row_stride,
                );
                blas::gemv::notrans(
                    n_rows - i - 1,
                    i,
                    -T::one(),
                    a.offset((i + 1) as isize * a_row_stride),
                    a_row_stride,
                    a_col_stride,
                    x.offset(i as isize * x_col_stride),
                    x_row_stride,
                    T::one(),
                    x.offset((i + 1) as isize * x_row_stride + i as isize * x_col_stride),
                    x_row_stride,
                );
                blas::gemv::notrans(
                    i,
                    n_cols - i,
                    T::one(),
                    a.offset(i as isize * a_col_stride),
                    a_row_stride,
                    a_col_stride,
                    a.offset(i as isize * a_row_stride + i as isize * a_col_stride),
                    a_col_stride,
                    T::zero(),
                    x.offset(i as isize * x_col_stride),
                    y_row_stride,
                );
                blas::gemv::notrans(
                    n_rows - i - 1,
                    i,
                    -T::one(),
                    x.offset((i + 1) as isize * x_row_stride),
                    x_row_stride,
                    x_col_stride,
                    x.offset(i as isize * x_col_stride),
                    x_row_stride,
                    T::one(),
                    x.offset((i + 1) as isize * x_row_stride + i as isize * x_col_stride),
                    x_row_stride,
                );
                blas::scal(
                    *tau_p.add(i),
                    &mut ArrayViewMut::from_shape_ptr(
                        [n_rows - i - 1].strides([x_row_stride as usize]),
                        x.offset((i + 1) as isize * x_row_stride + i as isize * x_col_stride),
                    ),
                );

                blas::gemv::notrans(
                    n_rows - i - 1,
                    i,
                    -T::one(),
                    a.offset((i + 1) as isize * a_row_stride),
                    a_row_stride,
                    a_col_stride,
                    y.offset(i as isize * y_row_stride),
                    y_col_stride,
                    T::one(),
                    a.offset((i + 1) as isize * a_row_stride + i as isize * a_col_stride),
                    a_row_stride,
                );
                blas::gemv::notrans(
                    n_rows - i + 1,
                    i + 1,
                    -T::one(),
                    x.offset((i + 1) as isize * x_row_stride),
                    x_row_stride,
                    x_col_stride,
                    a.offset(i as isize * a_col_stride),
                    a_row_stride,
                    T::one(),
                    a.offset((i + 1) as isize * a_row_stride + i as isize * a_col_stride),
                    a_row_stride,
                );

                let (beta, _, tau) = lapack::larfg(
                    *a.offset((i + 1) as isize * a_row_stride + i as isize * a_col_stride),
                    ArrayViewMut::from_shape_ptr(
                        [n_rows - i - 1].strides([a_row_stride as usize]),
                        a.offset(
                            cmp::min(i + 3, n_rows - 1) as isize * a_row_stride
                                + i as isize * a_col_stride,
                        ),
                    ),
                );
                *tau_q.add(i) = tau;
                *e.add(i) = beta.into();
                *a.offset((i + 1) as isize * a_row_stride + i as isize * a_col_stride) = T::one();

                blas::gemv::conjtrans(
                    n_rows - i - 1,
                    n_cols - i - 1,
                    T::one(),
                    a.offset((i + 1) as isize * a_row_stride + (i + 1) as isize * a_col_stride),
                    a_row_stride,
                    a_col_stride,
                    a.offset((i + 1) as isize * a_row_stride + i as isize * a_col_stride),
                    a_row_stride,
                    T::zero(),
                    y.offset((i + 1) as isize * y_row_stride + i as isize * y_col_stride),
                    y_row_stride,
                );
                blas::gemv::conjtrans(
                    n_rows - i - 1,
                    i,
                    T::one(),
                    a.offset((i + 1) as isize * a_row_stride),
                    a_row_stride,
                    a_col_stride,
                    a.offset((i + 1) as isize * a_row_stride + i as isize * a_col_stride),
                    a_row_stride,
                    T::zero(),
                    y.offset(i as isize * y_col_stride),
                    y_row_stride,
                );
                blas::gemv::notrans(
                    n_cols - i - 1,
                    i,
                    -T::one(),
                    y.offset((i + 1) as isize * y_row_stride),
                    y_row_stride,
                    y_col_stride,
                    y.offset(i as isize * y_col_stride),
                    y_row_stride,
                    T::one(),
                    y.offset((i + 1) as isize * y_row_stride + i as isize * y_col_stride),
                    y_row_stride,
                );
                blas::gemv::conjtrans(
                    n_rows - i - 1,
                    i + 1,
                    T::one(),
                    x.offset((i + 1) as isize * x_row_stride),
                    x_row_stride,
                    x_col_stride,
                    a.offset((i + 1) as isize * a_row_stride + i as isize * a_col_stride),
                    a_row_stride,
                    T::zero(),
                    y.offset(i as isize * y_col_stride),
                    y_row_stride,
                );
                blas::gemv::conjtrans(
                    i + 1,
                    n_cols - i - 1,
                    -T::one(),
                    a.offset((i + 1) as isize * a_col_stride),
                    a_row_stride,
                    a_col_stride,
                    y.offset(i as isize * y_col_stride),
                    y_row_stride,
                    T::one(),
                    y.offset((i + 1) as isize * y_row_stride + i as isize * y_col_stride),
                    y_row_stride,
                );
                blas::scal(
                    *tau_q.add(i),
                    &mut ArrayViewMut::from_shape_ptr(
                        [n_cols - i - 1].strides([y_row_stride as usize]),
                        y.offset((i + 1) as isize * y_row_stride + i as isize * y_col_stride),
                    ),
                );
            }
        }
    }
}

#[cfg(test)]
mod test {
    use ndarray::{arr2, Array1, Array2, Axis};

    #[test]
    fn wide() {
        let width = 1;
        let mut a = arr2(&[[2_f64, 1_f64, 2_f64], [1_f64, 0_f64, -3_f64]]);
        let mut d = Array1::<f64>::zeros(width);
        let mut e = Array1::<f64>::zeros(width);
        let mut tau_q = Array1::<f64>::zeros(width);
        let mut tau_p = Array1::<f64>::zeros(width);
        let mut x = Array2::<f64>::zeros((a.nrows(), width));
        let mut y = Array2::<f64>::zeros((a.ncols(), width));
        unsafe {
            super::labrd(
                a.nrows(),
                a.ncols(),
                width,
                a.as_mut_ptr(),
                a.stride_of(Axis(0)),
                a.stride_of(Axis(1)),
                d.as_mut_ptr(),
                e.as_mut_ptr(),
                tau_q.as_mut_ptr(),
                tau_p.as_mut_ptr(),
                x.as_mut_ptr(),
                width as isize,
                1,
                y.as_mut_ptr(),
                width as isize,
                1,
            )
        };
        println!("{}", a);
        println!("{}", x);
        println!("{}", y);
    }
}
