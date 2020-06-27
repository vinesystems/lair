use crate::{blas, lapack, Scalar};
use ndarray::{s, ArrayBase, Axis, Data, DataMut, Ix1, Ix2};

/// Applies an elementary reflector to a matrix.
///
/// # Panics
///
/// Panics if `v` is a zero vector, or `c` is a zero matrix.
pub fn left<A, SV, SC>(v: &ArrayBase<SV, Ix1>, tau: A, c: &mut ArrayBase<SC, Ix2>)
where
    A: Scalar,
    SV: Data<Elem = A>,
    SC: DataMut<Elem = A>,
{
    if tau == A::zero() {
        return;
    }
    let (last_v, _) = v
        .iter()
        .enumerate()
        .rev()
        .find(|(_, &elem)| elem != A::zero())
        .unwrap();
    let last_c = if let Some(last_c) = lapack::ilalc(&c.slice(s![0..=last_v, ..])) {
        last_c
    } else {
        return;
    };
    let w = blas::gemv_transpose(
        A::one(),
        &c.slice(s![0..=last_v, 0..=last_c]),
        &v.slice(s![0..=last_v]),
    );
    unsafe {
        blas::gerc(
            last_v + 1,
            last_c + 1,
            -tau,
            v.as_ptr(),
            v.stride_of(Axis(0)),
            w.as_ptr(),
            w.stride_of(Axis(0)),
            c.slice_mut(s![0..=last_v, 0..=last_c]).as_mut_ptr(),
            c.stride_of(Axis(0)),
            c.stride_of(Axis(1)),
        )
    };
}

/// Applies an elementary reflector to a matrix.
///
/// # Panics
///
/// Panics if `v` is a zero vector, or `c` is a zero matrix.
#[allow(dead_code)]
pub fn right<A, SV, SC>(v: &ArrayBase<SV, Ix1>, tau: A, c: &mut ArrayBase<SC, Ix2>)
where
    A: Scalar,
    SV: Data<Elem = A>,
    SC: DataMut<Elem = A>,
{
    if tau == A::zero() {
        return;
    }
    let (last_v, _) = v
        .iter()
        .enumerate()
        .rev()
        .find(|(_, &elem)| elem != A::zero())
        .unwrap();
    let last_r = if let Some(last_r) = lapack::ilalr(&c.slice(s![.., 0..=last_v])) {
        last_r
    } else {
        return;
    };
    let w = blas::gemv(
        A::one(),
        &c.slice(s![0..=last_r, 0..=last_v]),
        &v.slice(s![0..=last_v]),
    );
    unsafe {
        blas::gerc(
            last_r + 1,
            last_v + 1,
            -tau,
            w.as_ptr(),
            w.stride_of(Axis(0)),
            v.as_ptr(),
            v.stride_of(Axis(0)),
            c.slice_mut(s![0..=last_r, 0..=last_v]).as_mut_ptr(),
            c.stride_of(Axis(0)),
            c.stride_of(Axis(1)),
        )
    };
}

#[cfg(test)]
mod test {
    use approx::{assert_abs_diff_eq, AbsDiffEq};
    use ndarray::{arr1, arr2};
    use num_complex::Complex64;

    #[test]
    fn left_real() {
        let v = arr1(&[1., 2., 3.]);
        let mut c = arr2(&[[1., 2.], [3., 4.], [5., 6.]]);
        super::left(&v, 2., &mut c);
        assert!(c.abs_diff_eq(&arr2(&[[-43., -54.], [-85., -108.], [-127., -162.]]), 1e-8));
    }

    #[test]
    fn left_complex() {
        let v = arr1(&[
            Complex64::new(1., 3.),
            Complex64::new(2., 2.),
            Complex64::new(3., 1.),
        ]);
        let mut c = arr2(&[
            [Complex64::new(1., 2.), Complex64::new(2., 1.)],
            [Complex64::new(3., 4.), Complex64::new(4., 3.)],
            [Complex64::new(5., 6.), Complex64::new(6., 5.)],
        ]);
        super::left(&v, Complex64::new(2., 1.), &mut c);
        assert_abs_diff_eq!(c[(0, 0)].re, 141.);
        assert_abs_diff_eq!(c[(0, 0)].im, -278.);
        assert_abs_diff_eq!(c[(0, 1)].re, 58.);
        assert_abs_diff_eq!(c[(0, 1)].im, -291.);
        assert_abs_diff_eq!(c[(1, 0)].re, 3.);
        assert_abs_diff_eq!(c[(1, 0)].im, -276.);
        assert_abs_diff_eq!(c[(1, 1)].re, -68.);
        assert_abs_diff_eq!(c[(1, 1)].im, -253.);
        assert_abs_diff_eq!(c[(2, 0)].re, -135.);
        assert_abs_diff_eq!(c[(2, 0)].im, -274.);
        assert_abs_diff_eq!(c[(2, 1)].re, -194.);
        assert_abs_diff_eq!(c[(2, 1)].im, -215.);
    }

    #[test]
    fn right_real() {
        let v = arr1(&[1., 2., 3.]);
        let mut c = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
        super::right(&v, 2., &mut c);
        assert!(c.abs_diff_eq(&arr2(&[[-27., -54., -81.], [-60., -123., -186.]]), 1e-8));
    }

    #[test]
    fn right_complex() {
        let v = arr1(&[
            Complex64::new(1., 3.),
            Complex64::new(2., 2.),
            Complex64::new(3., 1.),
        ]);
        let mut c = arr2(&[
            [
                Complex64::new(1., 2.),
                Complex64::new(2., 1.),
                Complex64::new(3., 4.),
            ],
            [
                Complex64::new(4., 3.),
                Complex64::new(5., 6.),
                Complex64::new(6., 5.),
            ],
        ]);
        super::right(&v, Complex64::new(2., 1.), &mut c);
        assert_abs_diff_eq!(c[(0, 0)].re, -139.);
        assert_abs_diff_eq!(c[(0, 0)].im, -118.);
        assert_abs_diff_eq!(c[(0, 1)].re, -62.);
        assert_abs_diff_eq!(c[(0, 1)].im, -151.);
        assert_abs_diff_eq!(c[(0, 2)].re, 15.);
        assert_abs_diff_eq!(c[(0, 2)].im, -180.);
        assert_abs_diff_eq!(c[(1, 0)].re, -316.);
        assert_abs_diff_eq!(c[(1, 0)].im, -257.);
        assert_abs_diff_eq!(c[(1, 1)].re, -147.);
        assert_abs_diff_eq!(c[(1, 1)].im, -330.);
        assert_abs_diff_eq!(c[(1, 2)].re, 22.);
        assert_abs_diff_eq!(c[(1, 2)].im, -407.);
    }
}
