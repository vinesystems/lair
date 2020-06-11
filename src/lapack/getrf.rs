use ndarray::{ArrayBase, DataMut, Ix2, NdFloat};
use std::cmp;

#[derive(Debug)]
pub(crate) struct Singular();

pub(crate) fn getrf<A, S>(a: &mut ArrayBase<S, Ix2>) -> Result<Vec<usize>, Singular>
where
    A: NdFloat,
    S: DataMut<Elem = A>,
{
    let mut p = (0..a.nrows()).collect::<Vec<_>>();
    for i in 0..cmp::min(a.nrows(), a.ncols()) {
        let mut abs_max = A::zero();
        let mut max_row = i;
        for k in i..a.nrows() {
            let abs = a[(k, i)].abs();
            if abs > abs_max {
                abs_max = abs;
                max_row = k;
            }
        }
        if abs_max == A::zero() {
            return Err(Singular());
        }

        if max_row != i {
            p.swap(i, max_row);

            for j in 0..a.ncols() {
                let max_val = a[(max_row, j)];
                let i_val = std::mem::replace(&mut a[(i, j)], max_val);
                a[(max_row, j)] = i_val;
            }
        }

        let pivot = a[(i, i)];
        for j in i + 1..a.nrows() {
            let ratio = a[(j, i)] / pivot;
            a[(j, i)] = ratio;
            for k in i + 1..a.ncols() {
                let elem = ratio * a[(i, k)];
                a[(j, k)] -= elem;
            }
        }
    }
    Ok(p)
}

#[cfg(test)]
mod test {
    use approx::{assert_relative_eq, relative_eq};
    use ndarray::{arr2, Array2, ArrayBase};

    #[test]
    fn singular() {
        let mut a = arr2(&[[1_f64, 1_f64], [1_f64, 1_f64]]);
        assert!(super::getrf(&mut a).is_err());
    }

    #[test]
    fn empty() {
        let mut a: Array2<f32> = ArrayBase::eye(0);
        let p = super::getrf(&mut a).expect("valid input");
        assert_eq!(p, vec![]);
    }

    #[test]
    fn smallest() {
        let mut a = arr2(&[[3_f64]]);
        let p = super::getrf(&mut a).expect("valid input");
        assert_eq!(p, vec![0]);
        assert_eq!(a, arr2(&[[3.]]))
    }

    #[test]
    fn square() {
        let mut a = arr2(&[
            [1_f64, 2_f64, 3_f64, 1_f64, 2_f64],
            [2_f64, 2_f64, 1_f64, 3_f64, 3_f64],
            [3_f64, 1_f64, 2_f64, 2_f64, 1_f64],
            [2_f64, 3_f64, 3_f64, 1_f64, 1_f64],
            [1_f64, 3_f64, 1_f64, 3_f64, 1_f64],
        ]);
        let p = super::getrf(&mut a).expect("valid input");
        assert_eq!(p, vec![2, 4, 0, 3, 1]);
        assert_relative_eq!(a[(0, 0)], 3.);
        assert_relative_eq!(a[(1, 0)], 0.3333333333333333);
        assert_relative_eq!(a[(2, 0)], 0.3333333333333333);
        assert_relative_eq!(a[(3, 0)], 0.6666666666666666);
        assert_relative_eq!(a[(4, 0)], 0.6666666666666666);
    }

    #[test]
    fn wide() {
        let mut a = arr2(&[[1_f64, 2_f64, 3_f64], [2_f64, 3_f64, 4_f64]]);
        let p = super::getrf(&mut a).expect("valid input");
        assert_eq!(p, vec![1, 0]);
        assert_eq!(a, arr2(&[[2., 3., 4.], [0.5, 0.5, 1.]]));
    }

    #[test]
    fn tall() {
        let mut a = arr2(&[[1_f64, 2_f64], [1_f64, 3_f64], [2_f64, 3_f64]]);
        let p = super::getrf(&mut a).expect("valid input");
        assert_eq!(p, vec![2, 1, 0]);
        let lu = arr2(&[[2., 3.], [0.5, 1.5], [0.5, 0.3333333333333333]]);
        assert!(a
            .iter()
            .zip(lu.iter())
            .all(|(a_elem, lu_elem)| relative_eq!(*a_elem, *lu_elem)));
    }
}
