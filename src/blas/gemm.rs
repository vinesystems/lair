use ndarray::{ArrayBase, Data, DataMut, Ix2};

use crate::Scalar;

/// Performs `c` += `alpha` * `a` * `b`.
pub fn gemm<A, SA, SB, SC>(
    alpha: A,
    a: &ArrayBase<SA, Ix2>,
    a_conjugate: bool,
    b: &ArrayBase<SB, Ix2>,
    b_conjugate: bool,
    c: &mut ArrayBase<SC, Ix2>,
) where
    A: Scalar,
    SA: Data<Elem = A>,
    SB: Data<Elem = A>,
    SC: DataMut<Elem = A>,
{
    for (a_row, c_row) in a.rows().into_iter().zip(c.rows_mut()) {
        for (b_col, c_elem) in b.columns().into_iter().zip(c_row.into_iter()) {
            let dot_product = a_row
                .iter()
                .zip(b_col)
                .fold(A::zero(), |sum, (a_elem, b_elem)| {
                    let a_elem = if a_conjugate { a_elem.conj() } else { *a_elem };
                    let b_elem = if b_conjugate { b_elem.conj() } else { *b_elem };
                    sum + a_elem * b_elem
                });
            *c_elem += alpha * dot_product;
        }
    }
}
