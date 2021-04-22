use crate::Scalar;
use ndarray::{ArrayBase, DataMut, Ix1, Ix2};

#[allow(dead_code)]
pub fn upper<A, SD, SE, SV, SU, SC>(
    d: &ArrayBase<SD, Ix1>,
    e: &ArrayBase<SE, Ix1>,
    vt: &ArrayBase<SV, Ix2>,
    u: &ArrayBase<SU, Ix2>,
    c: &ArrayBase<SC, Ix2>,
) where
    A: Scalar,
    SD: DataMut<Elem = A::Real>,
    SE: DataMut<Elem = A::Real>,
    SV: DataMut<Elem = A>,
    SU: DataMut<Elem = A>,
    SC: DataMut<Elem = A>,
{
    let (_, _) = (d, e);
    if vt.ncols() == 0 && u.nrows() == 0 && c.ncols() == 0 {
        // call slasq1
        todo!()
    }
    todo!()
}
