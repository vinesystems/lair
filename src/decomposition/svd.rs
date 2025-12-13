use std::{
    cmp, fmt,
    ops::{AddAssign, Div, Mul, MulAssign, Sub},
};

use ndarray::{Array1, Array2, ArrayBase, DataMut, Ix2};

use crate::{
    lapack::{gesdd, gesvd},
    Scalar,
};

/// An error which can be returned when a singular value decomposition fails to
/// converge.
///
/// It contains the number of superdiagonals that did not converge to zero.
#[derive(Debug)]
pub struct SvdError(usize);

impl fmt::Display for SvdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} superdiagonals did not converge to zero", self.0)
    }
}

impl std::error::Error for SvdError {}

pub type SvdOutput<A> = (Array2<A>, Array1<<A as Scalar>::Real>, Option<Array2<A>>);

/// Calls gesvd.
///
/// # Errors
///
/// Returns [`SvdError`] if the SVD algorithm failed to converge.
///
/// [`SvdError`]: ../struct.SvdError.html
///
/// # Panics
///
/// Panics if `a`'s memory layout is not contiguous or not in the standard
/// layout.
pub fn svd<A, S>(a: &mut ArrayBase<S, Ix2>, calc_vt: bool) -> Result<SvdOutput<A>, SvdError>
where
    A: Scalar
        + Div<<A as Scalar>::Real, Output = A>
        + MulAssign<<A as Scalar>::Real>
        + AddAssign
        + Mul<A::Real, Output = A>
        + Sub<Output = A>,
    S: DataMut<Elem = A>,
{
    assert!(a.is_standard_layout());
    let k = cmp::min(a.nrows(), a.ncols());
    let mut s = unsafe { Array1::uninit(k).assume_init() };
    let mut u = unsafe { Array2::uninit((a.nrows(), a.nrows())).assume_init() };
    let mut vt = if calc_vt {
        unsafe { Array2::uninit((a.ncols(), a.ncols())).assume_init() }
    } else {
        Array2::zeros((0, 0))
    };
    gesvd::gesvd(
        gesvd::JobU::All,
        if calc_vt {
            gesvd::JobVT::All
        } else {
            gesvd::JobVT::None
        },
        a,
        s.as_slice_mut().expect("standard layout"),
        &mut u,
        &mut vt,
    )
    .map_err(SvdError)?;
    Ok((u, s, if calc_vt { Some(vt) } else { None }))
}

pub type SvddcOutput<A> = (Array2<A>, Array1<<A as Scalar>::Real>, Array2<A>);

/// Calls gesdd.
///
/// # Errors
///
/// Returns [`SvdError`] if the SVD algorithm failed to converge.
///
/// [`SvdError`]: ../struct.SvdError.html
///
/// # Panics
///
/// Panics if `a`'s memory layout is not contiguous or not in the standard
/// layout.
pub fn svddc<A, S>(a: &mut ArrayBase<S, Ix2>) -> Result<SvddcOutput<A>, SvdError>
where
    A: Scalar
        + Div<<A as Scalar>::Real, Output = A>
        + MulAssign<<A as Scalar>::Real>
        + AddAssign
        + Mul<A::Real, Output = A>
        + Sub<Output = A>,
    S: DataMut<Elem = A>,
{
    assert!(a.is_standard_layout());
    let k = cmp::min(a.nrows(), a.ncols());
    let mut s = unsafe { Array1::uninit(k).assume_init() };
    let mut u = unsafe { Array2::uninit((a.nrows(), a.nrows())).assume_init() };
    let mut vt = unsafe { Array2::uninit((a.ncols(), a.ncols())).assume_init() };
    gesdd::gesdd(
        gesdd::Job::All,
        a,
        s.as_slice_mut().expect("standard layout"),
        &mut u,
        &mut vt,
    )
    .map_err(SvdError)?;
    Ok((u, s, vt))
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    use super::{svd, svddc, SvdError};

    #[test]
    fn svd_error_display() {
        let err = SvdError(5);
        assert_eq!(err.to_string(), "5 superdiagonals did not converge to zero");
    }

    #[test]
    fn svd_calc_vt_true() {
        let mut a = arr2(&[[1.0f64, 2.0], [3.0, 4.0]]);
        let result = svd(&mut a, true).unwrap();
        let (_u, _s, vt) = result;
        assert!(vt.is_some(), "vt should be Some when calc_vt is true");
        let vt = vt.unwrap();
        assert_eq!(vt.shape(), &[2, 2], "vt should have shape (nrows, nrows)");
    }

    #[test]
    fn svd_calc_vt_false() {
        let mut a = arr2(&[[1.0f64, 2.0], [3.0, 4.0]]);
        let result = svd(&mut a, false).unwrap();
        let (_u, _s, vt) = result;
        assert!(vt.is_none(), "vt should be None when calc_vt is false");
    }

    #[test]
    fn svd_square_matrix_shapes() {
        let mut a = arr2(&[[1.0f64, 2.0], [3.0, 4.0]]);
        let result = svd(&mut a, true).unwrap();
        let (u, s, vt) = result;
        assert_eq!(u.shape(), &[2, 2], "u should have shape (nrows, nrows)");
        assert_eq!(s.len(), 2, "s should have length min(nrows, ncols)");
        assert_eq!(
            vt.unwrap().shape(),
            &[2, 2],
            "vt should have shape (ncols, ncols)"
        );
    }

    #[test]
    fn svd_wide_matrix_shapes() {
        let mut a = arr2(&[[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let result = svd(&mut a, true).unwrap();
        let (u, s, vt) = result;
        assert_eq!(u.shape(), &[2, 2], "u should have shape (nrows, nrows)");
        assert_eq!(s.len(), 2, "s should have length min(nrows, ncols)");
        assert_eq!(
            vt.unwrap().shape(),
            &[3, 3],
            "vt should have shape (ncols, ncols)"
        );
    }

    #[test]
    fn svd_tall_matrix_shapes() {
        let mut a = arr2(&[[1.0f64, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let result = svd(&mut a, true).unwrap();
        let (u, s, vt) = result;
        assert_eq!(u.shape(), &[3, 3], "u should have shape (nrows, nrows)");
        assert_eq!(s.len(), 2, "s should have length min(nrows, ncols)");
        assert_eq!(
            vt.unwrap().shape(),
            &[2, 2],
            "vt should have shape (ncols, ncols)"
        );
    }

    #[test]
    fn svd_single_element() {
        let mut a = arr2(&[[3.0f64]]);
        let result = svd(&mut a, true).unwrap();
        let (u, s, vt) = result;
        assert_eq!(u.shape(), &[1, 1], "u should have shape (1, 1)");
        assert_eq!(s.len(), 1, "s should have length 1");
        assert_eq!(vt.unwrap().shape(), &[1, 1], "vt should have shape (1, 1)");
    }

    #[test]
    fn svd_calc_vt_false_wide_matrix() {
        let mut a = arr2(&[[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let result = svd(&mut a, false).unwrap();
        let (u, s, vt) = result;
        assert_eq!(u.shape(), &[2, 2], "u should have shape (nrows, nrows)");
        assert_eq!(s.len(), 2, "s should have length min(nrows, ncols)");
        assert!(vt.is_none(), "vt should be None when calc_vt is false");
    }

    #[test]
    fn svddc_always_returns_vt() {
        let mut a = arr2(&[[1.0f64, 2.0], [3.0, 4.0]]);
        let result = svddc(&mut a).unwrap();
        let (_u, _s, vt) = result;
        assert_eq!(vt.shape(), &[2, 2], "svddc should always return vt");
    }

    #[test]
    fn svddc_square_matrix_shapes() {
        let mut a = arr2(&[[1.0f64, 2.0], [3.0, 4.0]]);
        let result = svddc(&mut a).unwrap();
        let (u, s, vt) = result;
        assert_eq!(u.shape(), &[2, 2], "u should have shape (nrows, nrows)");
        assert_eq!(s.len(), 2, "s should have length min(nrows, ncols)");
        assert_eq!(vt.shape(), &[2, 2], "vt should have shape (ncols, ncols)");
    }

    #[test]
    fn svddc_wide_matrix_shapes() {
        let mut a = arr2(&[[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let result = svddc(&mut a).unwrap();
        let (u, s, vt) = result;
        assert_eq!(u.shape(), &[2, 2], "u should have shape (nrows, nrows)");
        assert_eq!(s.len(), 2, "s should have length min(nrows, ncols)");
        assert_eq!(vt.shape(), &[3, 3], "vt should have shape (ncols, ncols)");
    }

    #[test]
    fn svddc_tall_matrix_shapes() {
        let mut a = arr2(&[[1.0f64, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let result = svddc(&mut a).unwrap();
        let (u, s, vt) = result;
        assert_eq!(u.shape(), &[3, 3], "u should have shape (nrows, nrows)");
        assert_eq!(s.len(), 2, "s should have length min(nrows, ncols)");
        assert_eq!(vt.shape(), &[2, 2], "vt should have shape (ncols, ncols)");
    }

    #[test]
    fn svddc_single_element() {
        let mut a = arr2(&[[3.0f64]]);
        let result = svddc(&mut a).unwrap();
        let (u, s, vt) = result;
        assert_eq!(u.shape(), &[1, 1], "u should have shape (1, 1)");
        assert_eq!(s.len(), 1, "s should have length 1");
        assert_eq!(vt.shape(), &[1, 1], "vt should have shape (1, 1)");
    }
}
