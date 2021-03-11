use crate::Scalar;

#[allow(
    clippy::cast_possible_wrap,
    clippy::similar_names,
    clippy::too_many_arguments
)]
pub unsafe fn gerc<T>(
    nrows: usize,
    ncols: usize,
    alpha: T,
    x: *const T,
    incx: isize,
    y: *const T,
    incy: isize,
    a: *mut T,
    row_stride: isize,
    col_stride: isize,
) where
    T: Scalar,
{
    let mut xp = x;
    for i in 0..nrows {
        let mut ap = a.offset(row_stride * i as isize);
        let mut yp = y;
        let factor = alpha * *xp;
        for _ in 0..ncols {
            *ap += factor * (*yp).conj();
            yp = yp.offset(incy);
            ap = ap.offset(col_stride);
        }
        xp = xp.offset(incx);
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2, Axis};
    use num_complex::Complex32;

    #[test]
    fn complex() {
        let mut a = arr2(&[
            [Complex32::new(1., 2.), Complex32::new(2., 1.)],
            [Complex32::new(3., 4.), Complex32::new(4., 3.)],
            [Complex32::new(5., 6.), Complex32::new(6., 5.)],
        ]);
        let x = arr1(&[
            Complex32::new(1., 3.),
            Complex32::new(2., 2.),
            Complex32::new(3., 1.),
        ]);
        let y = arr1(&[Complex32::new(-1., -2.), Complex32::new(-3., -4.)]);
        unsafe {
            super::gerc(
                a.nrows(),
                a.ncols(),
                Complex32::new(1., -1.),
                x.as_ptr(),
                1,
                y.as_ptr(),
                1,
                a.as_mut_ptr(),
                a.stride_of(Axis(0)),
                a.stride_of(Axis(1)),
            )
        };
        assert_abs_diff_eq!(a[(0, 0)].re, -7.);
        assert_abs_diff_eq!(a[(0, 0)].im, 8.);
        assert_abs_diff_eq!(a[(0, 1)].re, -18.);
        assert_abs_diff_eq!(a[(0, 1)].im, 11.);
        assert_abs_diff_eq!(a[(1, 0)].re, -1.);
        assert_abs_diff_eq!(a[(1, 0)].im, 12.);
        assert_abs_diff_eq!(a[(1, 1)].re, -8.);
        assert_abs_diff_eq!(a[(1, 1)].im, 19.);
        assert_abs_diff_eq!(a[(2, 0)].re, 5.);
        assert_abs_diff_eq!(a[(2, 0)].im, 16.);
        assert_abs_diff_eq!(a[(2, 1)].re, 2.);
        assert_abs_diff_eq!(a[(2, 1)].im, 27.);
    }
}
