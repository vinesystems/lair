use crate::Scalar;
use ndarray::{ArrayBase, Data, DataMut, Ix2};

pub fn gemm<A, SA, SB, SC>(
    a: &ArrayBase<SA, Ix2>,
    b: &ArrayBase<SB, Ix2>,
    c: &mut ArrayBase<SC, Ix2>,
) where
    A: Scalar,
    SA: Data<Elem = A>,
    SB: Data<Elem = A>,
    SC: DataMut<Elem = A>,
{
    for (a_row, c_row) in a.rows().into_iter().zip(c.rows_mut()) {
        for (b_col, c_elem) in b.columns().into_iter().zip(c_row.into_iter()) {
            *c_elem -= a_row.dot(&b_col);
        }
    }
}
