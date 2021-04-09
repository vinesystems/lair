use crate::Scalar;
use ndarray::{ArrayBase, Data, DataMut, Ix2};
use std::ops::MulAssign;

#[allow(dead_code)]
pub fn right_lower_notrans<A, SA, SB, const UNIT: bool>(
    a: &ArrayBase<SA, Ix2>,
    b: &mut ArrayBase<SB, Ix2>,
) where
    A: Scalar,
    SA: Data<Elem = A>,
    SB: DataMut<Elem = A>,
{
    assert_eq!(b.ncols(), a.nrows());
    for (j, a_col) in a.columns().into_iter().enumerate() {
        if !UNIT {
            b.column_mut(j).mul_assign(a_col[j]);
        }
        for k in j + 1..b.ncols() {
            let multiplier = a_col[k];
            if multiplier == A::zero() {
                continue;
            }

            for i in 0..b.nrows() {
                let b_i_k = b[(i, k)];
                b[(i, j)] += multiplier * b_i_k;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    #[test]
    fn right_lower_notrans_nounit() {
        let a = arr2(&[[-1., 2.], [1., -3.]]);
        let mut b = arr2(&[[-2., 3.], [-3., 1.]]);
        super::right_lower_notrans::<_, _, _, false>(&a, &mut b);
        assert_eq!(b, arr2(&[[5., -9.], [4., -3.]]));
    }
}
