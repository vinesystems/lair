use crate::Scalar;
use ndarray::{ArrayBase, DataMut, Ix1};

pub fn scal<A, S>(a: A, x: &mut ArrayBase<S, Ix1>)
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    for elem in x.iter_mut() {
        *elem *= a;
    }
}

pub fn rscal<R, A, S>(a: R, x: &mut ArrayBase<S, Ix1>)
where
    R: Into<A::Real>,
    A: Scalar,
    S: DataMut<Elem = A>,
{
    let a = a.into();
    for elem in x.iter_mut() {
        *elem = elem.mul_real(a);
    }
}
