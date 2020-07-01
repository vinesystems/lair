use ndarray::{ArrayBase, DataMut, Ix1};
use std::ops::MulAssign;

pub fn scal<TA, TX, S>(a: TA, x: &mut ArrayBase<S, Ix1>)
where
    TA: Copy,
    TX: MulAssign<TA>,
    S: DataMut<Elem = TX>,
{
    for elem in x.iter_mut() {
        *elem *= a;
    }
}
