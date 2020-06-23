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
    let mut w = c
        .slice(s![0..=last_v, 0..=last_c])
        .t()
        .dot(&v.slice(s![0..=last_v]));
    blas::scal(tau, &mut w);
    let tau_w = w.insert_axis(Axis(0));
    let mut c_view = c.slice_mut(s![0..=last_v, 0..=last_c]);
    c_view -= &v.view().insert_axis(Axis(1)).dot(&tau_w);
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
    let mut w = c
        .slice(s![0..=last_r, 0..=last_v])
        .dot(&v.slice(s![0..=last_v]));
    blas::scal(tau, &mut w);
    let tau_w = w.insert_axis(Axis(1));
    let mut c_view = c.slice_mut(s![0..=last_r, 0..=last_v]);
    c_view -= &tau_w.dot(&v.view().insert_axis(Axis(0)));
}

#[cfg(test)]
mod test {
    use approx::AbsDiffEq;
    use ndarray::{arr1, arr2};

    #[test]
    fn left() {
        let v = arr1(&[1., 2., 3.]);
        let mut c = arr2(&[[1., 2.], [3., 4.], [5., 6.]]);
        super::left(&v, 2., &mut c);
        assert!(c.abs_diff_eq(&arr2(&[[-43., -54.], [-85., -108.], [-127., -162.]]), 1e-6));
    }

    #[test]
    fn right() {
        let v = arr1(&[1., 2., 3.]);
        let mut c = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
        super::right(&v, 2., &mut c);
        assert!(c.abs_diff_eq(&arr2(&[[-27., -54., -81.], [-60., -123., -186.]]), 1e-6));
    }
}
