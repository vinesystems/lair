use ndarray::{ArrayBase, DataMut, Ix2};
use num_traits::Float;
use std::ops::{Div, SubAssign};

pub(crate) fn getrf<A, S>(a: &mut ArrayBase<S, Ix2>) -> Vec<usize>
where
    A: Float + Div + SubAssign,
    S: DataMut<Elem = A>,
{
    let mut p = (0..a.nrows()).collect::<Vec<_>>();
    for i in 0..a.nrows() {
        let mut abs_max = A::zero();
        let mut max_row = i;
        for k in i..a.nrows() {
            let abs = a[(k, i)];
            if abs > abs_max {
                abs_max = abs;
                max_row = k;
            }
        }
        if abs_max == A::zero() {
            // singular
            return p;
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
            for k in i + 1..a.nrows() {
                let elem = ratio * a[(i, k)];
                a[(j, k)] -= elem;
            }
        }
    }
    p
}

#[cfg(test)]
mod test {
    use ndarray::arr2;

    #[test]
    fn getrf_2() {
        let mut a = arr2(&[[1_f64, 3_f64], [2_f64, 4_f64]]);
        let p = super::getrf(&mut a);
        assert_eq!(p, vec![1, 0]);
        assert_eq!(a, arr2(&[[2., 4.], [0.5, 1.]]))
    }

    #[test]
    fn getrf_5() {
        let mut a = arr2(&[
            [1_f64, 2_f64, 3_f64, 4_f64, 5_f64],
            [2_f64, 3_f64, 4_f64, 5_f64, 6_f64],
            [3_f64, 4_f64, 5_f64, 6_f64, 7_f64],
            [4_f64, 5_f64, 6_f64, 7_f64, 8_f64],
            [5_f64, 6_f64, 7_f64, 8_f64, 9_f64],
        ]);
        let p = super::getrf(&mut a);
        assert_eq!(p, vec![4, 0, 3, 2, 1]);
        assert_eq!(a[(0, 0)], 5.);
        assert_eq!(a[(1, 0)], 0.2);
        assert_eq!(a[(2, 0)], 0.8);
        assert_eq!(a[(3, 0)], 0.6);
        assert_eq!(a[(4, 0)], 0.4);
    }
}
