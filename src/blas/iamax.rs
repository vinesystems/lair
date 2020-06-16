use crate::Scalar;
use ndarray::{ArrayBase, Data, Ix1};

pub(crate) fn iamax<A, S>(x: &ArrayBase<S, Ix1>) -> (usize, A::Real)
where
    A: Scalar,
    S: Data<Elem = A>,
{
    let mut max_val = A::zero().re();
    let mut max_idx = 0;
    for (idx, elem) in x.iter().enumerate() {
        let val = elem.re().abs() + elem.im().abs();
        if val > max_val {
            max_val = val;
            max_idx = idx;
        }
    }
    (max_idx, max_val)
}

#[cfg(test)]
mod test {
    use ndarray::arr1;

    #[test]
    fn iamax() {
        let x = arr1(&[1., 3., 2.]);
        let (idx, max) = super::iamax(&x);
        assert_eq!(idx, 1);
        assert_eq!(max, 3.);
    }
}
