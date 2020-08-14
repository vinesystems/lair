use crate::{blas, Scalar};
use ndarray::{Array1, ArrayBase, Axis, Data, Ix1, Ix2};

/// Performs `y` = `alpha` * `a` * `x` + `beta` * `y`.
///
/// # Safety
///
/// * `a` is a two-dimensional array array with `n_rows` rows and `n_cols`
///   columns, with strides of `row_stride` and `col_stride`, respectively.
/// * `x` is an array of `n_cols` elements with stride of `inc_x`.
/// * `y` is an array of `n_rows` elements with stride of `inc_y`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemv<T>(
    n_rows: usize,
    n_cols: usize,
    alpha: T,
    a: *const T,
    row_stride: isize,
    col_stride: isize,
    x: *const T,
    inc_x: isize,
    beta: T,
    y: *mut T,
    inc_y: isize,
) where
    T: Scalar,
{
    if beta != T::one() {
        let mut y_elem = y;
        if beta == T::zero() {
            for _ in 0..n_rows {
                *y_elem = T::zero();
                y_elem = y_elem.offset(inc_y as isize);
            }
        } else {
            for _ in 0..n_rows {
                *y_elem *= beta;
                y_elem = y_elem.offset(inc_y as isize);
            }
        }
    }
    if alpha == T::zero() {
        return;
    }

    let mut a_col = a;
    let mut x_elem = x;
    for _ in 0..n_cols {
        let mut a_elem = a_col;
        let mut y_elem = y;
        let alpha_x = alpha * *x_elem;
        for _ in 0..n_rows {
            *y_elem += alpha_x * *a_elem;
            a_elem = a_elem.offset(row_stride);
            y_elem = y_elem.offset(inc_y);
        }
        a_col = a_col.offset(col_stride);
        x_elem = x_elem.offset(inc_x);
    }
}

#[allow(
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_arguments
)]
pub(crate) unsafe fn gemv_sub<T>(
    m: usize,
    n: usize,
    a: *const T,
    row_stride: isize,
    col_stride: isize,
    x: *const T,
    incx: isize,
    y: *mut T,
    incy: isize,
) where
    T: Scalar,
{
    for i in 0..m {
        *y.offset(i as isize * incy) -=
            blas::dot(n, a.offset(i as isize * row_stride), col_stride, x, incx);
    }
}

#[allow(clippy::module_name_repetitions)]
pub fn gemv_transpose<A, SA, SX>(
    alpha: A,
    a: &ArrayBase<SA, Ix2>,
    x: &ArrayBase<SX, Ix1>,
) -> Array1<A>
where
    A: Scalar,
    SA: Data<Elem = A>,
    SX: Data<Elem = A>,
{
    debug_assert_eq!(a.nrows(), x.len());
    a.lanes(Axis(0))
        .into_iter()
        .map(|a_col| {
            alpha
                * a_col
                    .iter()
                    .zip(x.iter())
                    .map(|(&a_elem, &x_elem)| a_elem.conj() * x_elem)
                    .sum()
        })
        .collect()
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2, Axis};
    use num_complex::Complex32;

    #[test]
    fn gemv() {
        let y = unsafe {
            let mut a = arr2(&[
                [Complex32::new(1., 2.), Complex32::new(4., 3.)],
                [Complex32::new(2., 1.), Complex32::new(5., 6.)],
                [Complex32::new(3., 4.), Complex32::new(6., 5.)],
            ]);
            a.swap_axes(0, 1);
            let x = [
                Complex32::new(1., 3.),
                Complex32::new(2., 2.),
                Complex32::new(3., 1.),
            ];
            let mut y = [Complex32::new(0., 0.), Complex32::new(0., 0.)];
            super::gemv(
                a.nrows(),
                a.ncols(),
                Complex32::new(1., 1.),
                a.as_ptr(),
                a.stride_of(Axis(0)),
                a.stride_of(Axis(1)),
                x.as_ptr(),
                1,
                Complex32::new(0., 0.),
                y.as_mut_ptr(),
                1,
            );
            y
        };
        assert_abs_diff_eq!(y[(0)].re, -24.);
        assert_abs_diff_eq!(y[(0)].im, 28.);
        assert_abs_diff_eq!(y[(1)].re, -52.);
        assert_abs_diff_eq!(y[(1)].im, 64.);
    }

    #[test]
    fn gemv_transpose_complex() {
        let a = arr2(&[
            [Complex32::new(1., 2.), Complex32::new(2., 1.)],
            [Complex32::new(3., 4.), Complex32::new(4., 3.)],
            [Complex32::new(5., 6.), Complex32::new(6., 5.)],
        ]);
        let x = arr1(&[
            Complex32::new(1., 3.),
            Complex32::new(2., 2.),
            Complex32::new(3., 1.),
        ]);
        let y = super::gemv_transpose(Complex32::new(1., 1.), &a, &x);
        assert_abs_diff_eq!(y[(0)].re, 56.);
        assert_abs_diff_eq!(y[(0)].im, 28.);
        assert_abs_diff_eq!(y[(1)].re, 44.);
        assert_abs_diff_eq!(y[(1)].im, 40.);
    }
}
