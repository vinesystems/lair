use crate::Scalar;
use ndarray::{ArrayBase, Axis, Data, DataMut, Ix1, Ix2};

/// Performs `y` = `alpha` * `a` * `x` + `beta` * `y`.
pub fn notrans<A, SA, SX, SY>(
    alpha: A,
    a: &ArrayBase<SA, Ix2>,
    x: &ArrayBase<SX, Ix1>,
    beta: A,
    y: &mut ArrayBase<SY, Ix1>,
) where
    A: Scalar,
    SA: Data<Elem = A>,
    SX: Data<Elem = A>,
    SY: DataMut<Elem = A>,
{
    if beta == A::zero() {
        y.fill(A::zero());
    } else if beta != A::one() {
        *y *= beta;
    }
    if alpha == A::zero() {
        return;
    }

    for (a_col, &x_elem) in a.lanes(Axis(0)).into_iter().zip(x.iter()) {
        let alpha_x = alpha * x_elem;
        for (&a_elem, y_elem) in a_col.into_iter().zip(y.iter_mut()) {
            *y_elem += alpha_x * a_elem;
        }
    }
}

/// Performs `y` = `alpha` * `a.H` * `x` + `beta` * `y`.
///
/// # Safety
///
/// * `a` is a two-dimensional array array with `n_rows` rows and `n_cols`
///   columns, with strides of `row_stride` and `col_stride`, respectively.
/// * `x` is an array of `n_rows` elements with stride of `inc_x`.
/// * `y` is an array of `n_cols` elements with stride of `inc_y`.
pub fn conjtrans<A, SA, SX, SY>(
    alpha: A,
    a: &ArrayBase<SA, Ix2>,
    x: &ArrayBase<SX, Ix1>,
    beta: A,
    y: &mut ArrayBase<SY, Ix1>,
) where
    A: Scalar,
    SA: Data<Elem = A>,
    SX: Data<Elem = A>,
    SY: DataMut<Elem = A>,
{
    if beta == A::zero() {
        y.fill(A::zero());
    } else if beta != A::one() {
        *y *= beta;
    }
    if alpha == A::zero() {
        return;
    }

    for (a_col, y_elem) in a.lanes(Axis(0)).into_iter().zip(y.iter_mut()) {
        let sum = a_col
            .into_iter()
            .zip(x.iter())
            .fold(A::zero(), |sum, (&a_elem, &x_elem)| {
                sum + a_elem.conj() * x_elem
            });
        *y_elem += alpha * sum;
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2};
    use num_complex::Complex32;

    #[test]
    fn gemv_notrans() {
        let a = arr2(&[
            [
                Complex32::new(1., 2.),
                Complex32::new(2., 1.),
                Complex32::new(3., 4.),
            ],
            [
                Complex32::new(4., 3.),
                Complex32::new(5., 6.),
                Complex32::new(6., 5.),
            ],
        ]);
        let x = arr1(&[
            Complex32::new(1., 3.),
            Complex32::new(2., 2.),
            Complex32::new(3., 1.),
        ]);
        let mut y = arr1(&[Complex32::new(0., 0.), Complex32::new(0., 0.)]);
        super::notrans(
            Complex32::new(1., 1.),
            &a,
            &x,
            Complex32::new(0., 0.),
            &mut y,
        );
        assert_abs_diff_eq!(y[(0)].re, -24.);
        assert_abs_diff_eq!(y[(0)].im, 28.);
        assert_abs_diff_eq!(y[(1)].re, -52.);
        assert_abs_diff_eq!(y[(1)].im, 64.);
    }

    #[test]
    fn gemv_conjtrans_complex() {
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
        let mut y = arr1(&[Complex32::new(0., 0.), Complex32::new(0., 0.)]);
        super::conjtrans(
            Complex32::new(1., 1.),
            &a,
            &x,
            Complex32::new(0., 0.),
            &mut y,
        );
        assert_abs_diff_eq!(y[(0)].re, 56.);
        assert_abs_diff_eq!(y[(0)].im, 28.);
        assert_abs_diff_eq!(y[(1)].re, 44.);
        assert_abs_diff_eq!(y[(1)].im, 40.);
    }
}
