use crate::{blas, lapack, Scalar};
use ndarray::{s, ArrayBase, Axis, Data, DataMut, Ix1, Ix2};

#[allow(dead_code)]
pub fn ungrq<A, SA, ST>(a: &mut ArrayBase<SA, Ix2>, tau: &ArrayBase<ST, Ix1>)
where
    A: Scalar,
    SA: DataMut<Elem = A>,
    ST: Data<Elem = A>,
{
    ungr2(a, tau)
}

fn ungr2<A, SA, ST>(a: &mut ArrayBase<SA, Ix2>, tau: &ArrayBase<ST, Ix1>)
where
    A: Scalar,
    SA: DataMut<Elem = A>,
    ST: Data<Elem = A>,
{
    assert!(a.ncols() >= a.nrows());
    assert!(a.nrows() >= tau.len());

    let a_nrows = a.nrows();
    let a_ncols = a.ncols();
    if tau.len() < a_nrows {
        for j in 0..a_ncols {
            for l in 0..a_nrows - tau.len() {
                a[(l, j)] = A::zero();
            }
            if j >= a_ncols - a_nrows && j < a_ncols - tau.len() {
                a[(j + a_nrows - a_ncols, j)] = A::one();
            }
        }
    }
    for (i, &tau_i) in tau.iter().enumerate() {
        let (mut upper, mut lower) = a
            .slice_mut(s![
                ..=a.nrows() - tau.len() + i,
                ..=a.ncols() - tau.len() + i
            ])
            .split_at(Axis(0), a_nrows - tau.len() + i);
        let mut row = lower.row_mut(0);
        lapack::lacgv(&mut row);
        let width = row.len();
        row[width - 1] = A::one();
        lapack::larf::right(&row, tau_i.conj(), &mut upper);
        blas::scal(-tau_i, &mut row);
        lapack::lacgv(&mut row);
        row[width - 1] = A::one() - tau_i.conj();

        a.slice_mut(s![i, a_ncols - tau.len() + i + 1..])
            .fill(A::zero());
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};
    use num_complex::Complex32;

    #[test]
    fn ungrq() {
        let mut a = arr2(&[
            [
                Complex32::new(2., 1.),
                Complex32::new(3., -2.),
                Complex32::new(1., 3.),
            ],
            [
                Complex32::new(-1., -2.),
                Complex32::new(-3., 1.),
                Complex32::new(-2., -1.),
            ],
        ]);
        let tau = arr1(&[Complex32::new(1., -1.)]);
        super::ungrq(&mut a, &tau);
        assert_eq!(
            a,
            arr2(&[
                [
                    Complex32::new(6., -8.),
                    Complex32::new(-9., -10.),
                    Complex32::new(2., 4.)
                ],
                [
                    Complex32::new(-1., 3.),
                    Complex32::new(4., 2.),
                    Complex32::new(0., -1.)
                ]
            ])
        );
    }
}
