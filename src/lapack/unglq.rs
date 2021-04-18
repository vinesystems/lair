use crate::{blas, lapack, Scalar};
use ndarray::{s, ArrayBase, Axis, Data, DataMut, Ix1, Ix2};

#[allow(dead_code)]
pub fn unglq<A, SA, ST>(a: &mut ArrayBase<SA, Ix2>, tau: &ArrayBase<ST, Ix1>)
where
    A: Scalar,
    SA: DataMut<Elem = A>,
    ST: Data<Elem = A>,
{
    ungl2(a, tau)
}

fn ungl2<A, SA, ST>(a: &mut ArrayBase<SA, Ix2>, tau: &ArrayBase<ST, Ix1>)
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
            for l in tau.len()..a_nrows {
                a[(l, j)] = A::zero();
            }
            if j >= tau.len() && j < a_nrows {
                a[(j, j)] = A::one();
            }
        }
    }

    for (i, &tau_i) in tau.iter().enumerate().rev() {
        let (mut upper, mut lower) = a.slice_mut(s![i.., i..]).split_at(Axis(0), 1);
        let mut row = upper.row_mut(0);
        if i < a_ncols {
            lapack::lacgv(&mut row);
            if i < a_nrows {
                row[0] = A::one();
                lapack::larf::right(&row, tau_i.conj(), &mut lower);
            }
            blas::scal(-tau_i, &mut row);
            lapack::lacgv(&mut row);
        }
        row[0] = A::one() - tau_i.conj();
        a.slice_mut(s![i, ..i]).fill(A::zero());
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};
    use num_complex::Complex32;

    #[test]
    fn unglq() {
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
        super::unglq(&mut a, &tau);
        assert_eq!(
            a,
            arr2(&[
                [
                    Complex32::new(0., -1.),
                    Complex32::new(-5., -1.),
                    Complex32::new(2., -4.)
                ],
                [
                    Complex32::new(-1., -5.),
                    Complex32::new(-12., -13.),
                    Complex32::new(14., -8.)
                ],
            ])
        );
    }
}
