use crate::Scalar;
use ndarray::{Array1, ArrayBase, Axis, Data, Ix1, Ix2};

pub fn gemv<A, SA, SX>(alpha: A, a: &ArrayBase<SA, Ix2>, x: &ArrayBase<SX, Ix1>) -> Array1<A>
where
    A: Scalar,
    SA: Data<Elem = A>,
    SX: Data<Elem = A>,
{
    debug_assert_eq!(a.ncols(), x.len());
    a.lanes(Axis(1))
        .into_iter()
        .map(|a_row| {
            alpha
                * a_row
                    .iter()
                    .zip(x.iter())
                    .map(|(&a_elem, &x_elem)| a_elem * x_elem)
                    .sum()
        })
        .collect()
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
    use ndarray::{arr1, arr2};
    use num_complex::Complex32;

    #[test]
    fn gemv() {
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
        let y = super::gemv(Complex32::new(1., 1.), &a, &x);
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
