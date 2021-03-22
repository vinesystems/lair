use crate::Scalar;
use ndarray::{s, ArrayBase, Axis, DataMut, Ix2};
use std::cmp;

#[allow(dead_code)]
pub fn lower_zero<A, S>(a: &mut ArrayBase<S, Ix2>)
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    if a.ncols() == 0 {
        return;
    }

    let col_max = a.ncols() - 1;
    for (i, mut row) in a.lanes_mut(Axis(1)).into_iter().enumerate() {
        for v in row.slice_mut(s![..=cmp::min(i, col_max)]) {
            *v = A::zero();
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr2, Array2};

    #[test]
    fn lower_zero() {
        let mut a = Array2::<f64>::ones((0, 0));
        super::lower_zero(&mut a);
        assert_eq!(a, Array2::<f64>::zeros((0, 0)));

        let mut a = Array2::<f64>::ones((0, 2));
        super::lower_zero(&mut a);
        assert_eq!(a, Array2::<f64>::zeros((0, 2)));

        let mut a = Array2::<f64>::ones((2, 0));
        super::lower_zero(&mut a);
        assert_eq!(a, Array2::<f64>::zeros((2, 0)));

        let mut a = Array2::<f64>::ones((2, 3));
        super::lower_zero(&mut a);
        assert_eq!(a, arr2(&[[0., 1., 1.], [0., 0., 1.]]));

        let mut a = Array2::<f64>::ones((3, 2));
        super::lower_zero(&mut a);
        assert_eq!(a, arr2(&[[0., 1.], [0., 0.], [0., 0.]]));
    }
}
