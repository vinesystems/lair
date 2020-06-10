pub(crate) mod lapack;

use ndarray::{Array1, ArrayBase, Data, Ix1, Ix2, NdFloat};

pub fn solve<A, SA, SB>(a: &ArrayBase<SA, Ix2>, b: &ArrayBase<SB, Ix1>) -> Array1<A>
where
    A: NdFloat,
    SA: Data<Elem = A>,
    SB: Data<Elem = A>,
{
    let mut a = a.to_owned();
    let p = lapack::getrf(&mut a);
    lapack::getrs(&a, &p, b)
}
