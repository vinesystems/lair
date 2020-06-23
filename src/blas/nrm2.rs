use crate::Scalar;
use ndarray::{ArrayBase, Data, Ix1};

pub(crate) fn nrm2<A, S>(x: &ArrayBase<S, Ix1>) -> A::Real
where
    A: Scalar,
    S: Data<Elem = A>,
{
    x.iter()
        .map(|v| v.re() * v.re() + v.im() * v.im())
        .sum::<A::Real>()
        .sqrt()
}

#[cfg(test)]
mod test {
    use ndarray::arr1;

    #[test]
    fn nrm2() {
        let x = arr1(&[3., 4.]);
        let norm = super::nrm2(&x);
        assert_eq!(norm, 5.);
    }
}
