use ndarray::{s, ArrayBase, Axis, Data, DataMut, Ix1, Ix2};

use crate::{blas, lapack, Scalar};

/// Generates a matrix Q with orthonormal columns.
///
/// # Panics
///
/// Panics if `a` has more columns than rows, or has fewer columns than the
/// number of elementary reflectors.
#[allow(dead_code)]
pub fn ungqr<A, SA, ST>(a: &mut ArrayBase<SA, Ix2>, tau: &ArrayBase<ST, Ix1>)
where
    A: Scalar,
    SA: DataMut<Elem = A>,
    ST: Data<Elem = A>,
{
    ung2r(a, tau);
}

/// Generates all or part of the orthogonal matrix Q from a QR factorization.
///
/// # Panics
///
/// Panics if `a` has more columns than rows, or has fewer columns than the
/// number of elementary reflectors.
fn ung2r<A, SA, ST>(a: &mut ArrayBase<SA, Ix2>, tau: &ArrayBase<ST, Ix1>)
where
    A: Scalar,
    SA: DataMut<Elem = A>,
    ST: Data<Elem = A>,
{
    assert!(a.ncols() <= a.nrows(), "too many columns in `a`");
    assert!(tau.len() <= a.ncols(), "too many reflectors");

    if a.is_empty() {
        return;
    }

    a.slice_mut(s![.., tau.len()..]).fill(A::zero());
    for i in tau.len()..a.ncols() {
        *a.get_mut((i, i)).expect("valid index") = A::one();
    }

    for (i, tau_i) in tau.iter().enumerate().rev() {
        if i < a.ncols() - 1 {
            *a.get_mut((i, i)).expect("valid index") = A::one();
            let (v, mut c) = a.slice_mut(s![i.., i..]).split_at(Axis(1), 1);
            lapack::larf::left(&v.column(0), *tau_i, &mut c);
        }
        if i < a.nrows() - 1 {
            blas::scal(-*tau_i, &mut a.column_mut(i).slice_mut(s![i + 1..]));
        }
        *a.get_mut((i, i)).expect("valid index") = A::one() - *tau_i;
        if i >= 1 {
            a.slice_mut(s![..i, i]).fill(A::zero());
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};

    #[test]
    fn org2r() {
        let mut a = arr2(&[[2., -2.], [-1., 4.], [3., 1.]]);
        let tau = arr1(&[5., 3.]);
        super::ung2r(&mut a, &tau);
        assert_eq!(a, arr2(&[[-4., 35.], [5., -37.], [-15., 102.]]));
    }
}
