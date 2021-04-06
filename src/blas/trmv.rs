use crate::Scalar;
use ndarray::{ArrayBase, Axis, Data, DataMut, Ix1, Ix2};

/// Performs `x` = `a` * `x` for an upper-triangular matrix `a` and a column
/// vector `x`.
///
/// # Panics
///
/// Panics if `a`'s number of columns is not equal to `x`'s number of elements.
pub fn upper_notrans<A, SA, SX>(a: &ArrayBase<SA, Ix2>, x: &mut ArrayBase<SX, Ix1>)
where
    A: Scalar,
    SA: Data<Elem = A>,
    SX: DataMut<Elem = A>,
{
    assert_eq!(a.ncols(), x.len());
    for (j, a_col) in a.lanes(Axis(0)).into_iter().enumerate() {
        let multiplier = *unsafe { x.uget(j) };
        if multiplier == A::zero() {
            continue;
        }
        for (i, a_elem) in a_col.iter().take(j).enumerate() {
            *unsafe { x.uget_mut(i) } += multiplier * *a_elem;
        }
        *unsafe { x.uget_mut(j) } *= a_col[j];
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};

    #[test]
    fn unit() {
        let a = arr2(&[[1., 2.], [0., 1.]]);
        let mut x = arr1(&[-1., 1.]);
        super::upper_notrans(&a, &mut x);
        assert_eq!(x, arr1(&[1., 1.]));
    }
}
