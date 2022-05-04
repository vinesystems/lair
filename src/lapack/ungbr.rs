use crate::{lapack, Scalar};
use ndarray::{s, ArrayBase, Axis, Data, DataMut, Ix1, Ix2};

#[allow(dead_code)]
pub fn q_tall<A, SA, ST>(a: &mut ArrayBase<SA, Ix2>, tau: &ArrayBase<ST, Ix1>)
where
    A: Scalar,
    SA: DataMut<Elem = A>,
    ST: Data<Elem = A>,
{
    if a.is_empty() {
        return;
    }

    assert!(a.nrows() >= a.ncols());
    assert!(a.ncols() >= tau.len());
    lapack::ungqr(a, tau);
}

#[allow(dead_code)]
pub fn p_square<A, SA, ST>(a: &mut ArrayBase<SA, Ix2>, tau: &ArrayBase<ST, Ix1>)
where
    A: Scalar,
    SA: DataMut<Elem = A>,
    ST: Data<Elem = A>,
{
    if a.is_empty() {
        return;
    }

    assert_eq!(a.nrows(), a.ncols());
    assert_eq!(a.ncols(), tau.len());
    a.column_mut(0).fill(A::zero());
    a.row_mut(0)[0] = A::one();

    for (j, mut col) in a.lanes_mut(Axis(0)).into_iter().enumerate().skip(1) {
        for i in (1..j).rev() {
            col[i] = col[i - 1];
        }
        col[0] = A::zero();
    }
    if a.ncols() > 1 {
        lapack::unglq(&mut a.slice_mut(s![1.., 1..]), &tau.slice(s![..-1]));
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};
    use num_complex::Complex32;

    #[test]
    fn q_tall() {
        let mut a = arr2(&[
            [Complex32::new(2., 1.), Complex32::new(-3., 1.)],
            [Complex32::new(-1., -2.), Complex32::new(1., 3.)],
            [Complex32::new(3., -2.), Complex32::new(-2., -1.)],
        ]);
        let tau = arr1(&[Complex32::new(1., -1.)]);
        super::q_tall(&mut a, &tau);
        assert_eq!(
            a,
            arr2(&[
                [Complex32::new(0.0, 1.0), Complex32::new(-1.0, -3.0)],
                [Complex32::new(3.0, 1.0), Complex32::new(-4.0, 5.0)],
                [Complex32::new(-1.0, 5.0), Complex32::new(-9.0, -7.0)],
            ])
        );
    }

    #[test]
    fn p_square() {
        let mut a = arr2(&[
            [Complex32::new(2., 1.), Complex32::new(1., 3.)],
            [Complex32::new(-1., -2.), Complex32::new(-2., -1.)],
        ]);
        let tau = arr1(&[Complex32::new(1., -1.), Complex32::new(-2., 3.)]);
        super::p_square(&mut a, &tau);
        assert_eq!(
            a,
            arr2(&[
                [Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0)],
                [Complex32::new(0.0, 0.0), Complex32::new(0.0, -1.0),]
            ])
        );
    }
}
