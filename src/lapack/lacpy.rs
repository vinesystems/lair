use std::cmp;

use ndarray::{s, ArrayBase, Axis, Data, DataMut, Ix2};

use crate::Scalar;

#[allow(dead_code)]
pub fn lower<A, SA, SB>(a: &ArrayBase<SA, Ix2>, b: &mut ArrayBase<SB, Ix2>)
where
    A: Scalar,
    SA: Data<Elem = A>,
    SB: DataMut<Elem = A>,
{
    let ncols = cmp::min(a.ncols(), b.ncols());
    for (i, (a_row, mut b_row)) in a
        .lanes(Axis(1))
        .into_iter()
        .zip(b.lanes_mut(Axis(1)).into_iter())
        .enumerate()
    {
        let ncols = cmp::min(ncols, i + 1);
        for (a_v, b_v) in a_row
            .slice(s![..ncols])
            .iter()
            .zip(b_row.slice_mut(s![..ncols]).into_iter())
        {
            *b_v = *a_v;
        }
    }
}

#[allow(dead_code)]
pub fn upper<A, SA, SB>(a: &ArrayBase<SA, Ix2>, b: &mut ArrayBase<SB, Ix2>)
where
    A: Scalar,
    SA: Data<Elem = A>,
    SB: DataMut<Elem = A>,
{
    let col_min = cmp::min(a.ncols(), b.ncols());
    for (i, (a_row, mut b_row)) in a
        .lanes(Axis(1))
        .into_iter()
        .zip(b.lanes_mut(Axis(1)).into_iter())
        .enumerate()
        .take(col_min)
    {
        for (a_v, b_v) in a_row
            .slice(s![i..])
            .iter()
            .zip(b_row.slice_mut(s![i..]).into_iter())
        {
            *b_v = *a_v;
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr2, Array2};

    #[test]
    fn lower() {
        let a = Array2::<f64>::ones((0, 0));
        let mut b = Array2::<f64>::zeros((1, 2));
        super::lower(&a, &mut b);
        assert_eq!(b, Array2::<f64>::zeros((1, 2)));

        let a = Array2::<f64>::ones((1, 2));
        let mut b = Array2::<f64>::zeros((0, 0));
        super::lower(&a, &mut b);
        assert_eq!(b, Array2::<f64>::zeros((0, 0)));

        let a = Array2::<f64>::ones((2, 3));
        let mut b = Array2::<f64>::zeros((3, 2));
        super::lower(&a, &mut b);
        assert_eq!(b, arr2(&[[1., 0.], [1., 1.], [0., 0.]]));

        let a = Array2::<f64>::ones((3, 2));
        let mut b = Array2::<f64>::zeros((2, 3));
        super::lower(&a, &mut b);
        assert_eq!(b, arr2(&[[1., 0., 0.], [1., 1., 0.]]));
    }

    #[test]
    fn upper() {
        let a = Array2::<f64>::ones((0, 0));
        let mut b = Array2::<f64>::zeros((1, 2));
        super::upper(&a, &mut b);
        assert_eq!(b, Array2::<f64>::zeros((1, 2)));

        let a = Array2::<f64>::ones((1, 2));
        let mut b = Array2::<f64>::zeros((0, 0));
        super::upper(&a, &mut b);
        assert_eq!(b, Array2::<f64>::zeros((0, 0)));

        let a = Array2::<f64>::ones((2, 3));
        let mut b = Array2::<f64>::zeros((3, 2));
        super::upper(&a, &mut b);
        assert_eq!(b, arr2(&[[1., 1.], [0., 1.], [0., 0.]]));

        let a = Array2::<f64>::ones((3, 2));
        let mut b = Array2::<f64>::zeros((2, 3));
        super::upper(&a, &mut b);
        assert_eq!(b, arr2(&[[1., 1., 0.], [0., 1., 0.]]));
    }
}
