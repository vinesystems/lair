use crate::Scalar;
use ndarray::{ArrayBase, Axis, Data, DataMut, Ix1, Ix2};

pub fn gerc<A, SX, SY, SA>(
    alpha: A,
    x: &ArrayBase<SX, Ix1>,
    y: &ArrayBase<SY, Ix1>,
    a: &mut ArrayBase<SA, Ix2>,
) where
    A: Scalar,
    SX: Data<Elem = A>,
    SY: Data<Elem = A>,
    SA: DataMut<Elem = A>,
{
    for (&x_elem, mut a_row) in x.iter().zip(a.lanes_mut(Axis(1)).into_iter()) {
        for (&y_elem, a_elem) in y.iter().zip(a_row.iter_mut()) {
            *a_elem += alpha * x_elem * y_elem.conj();
        }
    }
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2};
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
        super::gerc(Complex32::new(1., -1.), &x, &y, &mut a);
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
