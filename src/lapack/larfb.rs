use ndarray::{Array2, ArrayBase, Axis, Data, DataMut, Ix2};

use crate::{blas, Scalar};

/// Applies a block reflector.
///
/// # Panics
///
/// Panics when
///
/// * `v` and `c` have different numbers of rows.
/// * `v` and `t` have different numbers of columns.
/// * `t` has more columns than the number of rows in `c`.
#[allow(dead_code)]
pub fn left_notrans_forward_columnwise<A, SV, ST, SC>(
    v: &ArrayBase<SV, Ix2>,
    t: &ArrayBase<ST, Ix2>,
    c: &mut ArrayBase<SC, Ix2>,
) where
    A: Scalar,
    SV: Data<Elem = A>,
    ST: Data<Elem = A>,
    SC: DataMut<Elem = A>,
{
    assert!(v.nrows() == c.nrows());
    assert!(v.ncols() == t.ncols());
    assert!(t.ncols() <= c.nrows());
    if c.is_empty() {
        return;
    }

    let c_nrows = c.nrows();
    let (c_upper, mut c_lower) = c.view_mut().split_at(Axis(0), t.ncols());
    let mut work = unsafe {
        // Will be initialized in the following `for` loop.
        Array2::<A>::uninit(c_upper.t().dim()).assume_init()
    };
    for (c_upper_row, work_col) in c_upper
        .lanes(Axis(1))
        .into_iter()
        .zip(work.lanes_mut(Axis(0)))
    {
        for (c_upper_elem, work_upper_elem) in c_upper_row.into_iter().zip(work_col) {
            *work_upper_elem = c_upper_elem.conj();
        }
    }
    let (v_upper, v_lower) = v.view().split_at(Axis(0), t.ncols());
    blas::trmm::right_lower_notrans::<_, _, _, true>(&v_upper, &mut work);
    if c_nrows > t.ncols() {
        blas::gemm(A::one(), &c_lower.t(), true, &v_lower, false, &mut work);
    }
    blas::trmm::right_upper_conjtrans::<_, _, _, false>(t, &mut work);
    if c_nrows > t.ncols() {
        blas::gemm(-A::one(), &v_lower, false, &work.t(), true, &mut c_lower);
    }
    blas::trmm::right_lower_conjtrans::<_, _, _, true>(&v_upper, &mut work);
    for (c_elem, work_elem) in c.iter_mut().zip(work.t()) {
        *c_elem -= work_elem.conj();
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    #[test]
    fn left_notrans_forward_columnwise() {
        let v = arr2(&[[1., 0.], [-1., 1.], [3., 2.]]);
        let t = arr2(&[[-2., 2.], [1., -3.]]);
        let mut c = arr2(&[[1., 4., 7.], [2., 5., 8.], [3., 6., 9.]]);
        super::left_notrans_forward_columnwise(&v, &t, &mut c);
        assert_eq!(c, arr2(&[[1., 4., 7.], [26., 56., 86.], [51., 108., 165.]]));
    }
}
