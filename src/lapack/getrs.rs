use crate::Scalar;
use ndarray::{Array1, ArrayBase, Axis, Data, Ix1, Ix2};

pub(crate) fn getrs<A, SA, SB>(
    a: &ArrayBase<SA, Ix2>,
    p: &[usize],
    b: &ArrayBase<SB, Ix1>,
) -> Array1<A>
where
    A: Scalar,
    SA: Data<Elem = A>,
    SB: Data<Elem = A>,
{
    assert_eq!(a.nrows(), p.len());
    let mut x = unsafe {
        let mut x: Array1<A> = ArrayBase::uninitialized(p.len());
        for (i, (row, idx)) in a.lanes(Axis(1)).into_iter().zip(p.iter()).enumerate() {
            x[i] = b[*idx];
            for (k, a_elem) in row.iter().take(i).enumerate() {
                let prod = *a_elem * x[k];
                x[i] -= prod;
            }
        }
        x
    };
    for i in (0..x.len()).rev() {
        for k in i + 1..x.len() {
            let prod = a[(i, k)] * x[k];
            x[i] -= prod;
        }
        x[i] /= a[(i, i)];
    }
    x
}

#[cfg(test)]
mod test {
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use ndarray::{arr1, arr2};

    #[test]
    fn square_2x2() {
        let mut a = arr2(&[[1_f64, 2_f64], [3_f64, 4_f64]]);
        let p = crate::lapack::getrf(&mut a).expect("valid input");
        assert_eq!(p, vec![1, 0]);
        let b = arr1(&[3_f64, 7_f64]);
        let x = super::getrs(&a, &p, &b);
        assert_relative_eq!(x[0], 1., max_relative = 1e-8);
        assert_relative_eq!(x[1], 1., max_relative = 1e-8);
    }

    #[test]
    fn square_5x5() {
        let mut a = arr2(&[
            [1_f64, 2_f64, 3_f64, 1_f64, 2_f64],
            [2_f64, 2_f64, 1_f64, 3_f64, 3_f64],
            [3_f64, 1_f64, 2_f64, 2_f64, 1_f64],
            [2_f64, 3_f64, 3_f64, 1_f64, 1_f64],
            [1_f64, 3_f64, 1_f64, 3_f64, 1_f64],
        ]);
        let p = crate::lapack::getrf(&mut a).expect("valid input");
        let b = arr1(&[9_f64, 11_f64, 9_f64, 10_f64, 9_f64]);
        let x = super::getrs(&a, &p, &b);
        assert_abs_diff_eq!(x[0], 1.);
        assert_abs_diff_eq!(x[1], 1.);
        assert_abs_diff_eq!(x[2], 1.);
        assert_abs_diff_eq!(x[3], 1.);
        assert_abs_diff_eq!(x[4], 1.);
    }
}
