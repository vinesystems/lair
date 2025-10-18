use ndarray::{Array1, ArrayBase, Axis, Data, Ix1, Ix2};

use crate::{lapack, Scalar};

/// Solves `a * x = b`.
///
/// # Panics
///
/// Panics if the number of rows in `a` is different from the number of elements
/// in `p` or the number of elements in `b`, or the number of columns in `a` is
/// smaller than the number of elements in `p`.
pub fn getrs<A, SA, SB>(a: &ArrayBase<SA, Ix2>, p: &[usize], b: &ArrayBase<SB, Ix1>) -> Array1<A>
where
    A: Scalar,
    SA: Data<Elem = A>,
    SB: Data<Elem = A>,
{
    assert_eq!(a.nrows(), p.len());
    assert_eq!(p.len(), b.len());
    assert!(a.ncols() >= p.len());
    unsafe {
        let mut x = b.to_owned();
        lapack::laswp(1, x.as_mut_ptr(), x.stride_of(Axis(0)), 1, 0, p);
        for (i, row) in a.lanes(Axis(1)).into_iter().enumerate() {
            for (k, a_elem) in row.iter().take(i).enumerate() {
                let prod = *a_elem * *x.uget(k);
                *x.uget_mut(i) -= prod;
            }
        }
        for i in (0..x.len()).rev() {
            for k in i + 1..x.len() {
                let prod = *a.uget((i, k)) * *x.uget(k);
                *x.uget_mut(i) -= prod;
            }
            *x.uget_mut(i) /= *a.uget((i, i));
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use ndarray::{arr1, arr2};

    #[test]
    fn square_2x2() {
        let x = {
            let mut a = arr2(&[[1_f64, 2_f64], [3_f64, 4_f64]]);
            let (p, _) = crate::lapack::getrf(a.view_mut());
            assert_eq!(p, vec![1, 1]);
            let b = arr1(&[3_f64, 7_f64]);
            super::getrs(&a, &p, &b)
        };
        assert_relative_eq!(x[0], 1., max_relative = 1e-8);
        assert_relative_eq!(x[1], 1., max_relative = 1e-8);
    }

    #[test]
    fn square_5x5() {
        let x = {
            let mut a = arr2(&[
                [1_f64, 2_f64, 3_f64, 1_f64, 2_f64],
                [2_f64, 2_f64, 1_f64, 3_f64, 3_f64],
                [3_f64, 1_f64, 2_f64, 2_f64, 1_f64],
                [2_f64, 3_f64, 3_f64, 1_f64, 1_f64],
                [1_f64, 3_f64, 1_f64, 3_f64, 1_f64],
            ]);
            let (p, _) = crate::lapack::getrf(a.view_mut());
            let b = arr1(&[9_f64, 11_f64, 9_f64, 10_f64, 9_f64]);
            super::getrs(&a, &p, &b)
        };
        assert_abs_diff_eq!(x[0], 1.);
        assert_abs_diff_eq!(x[1], 1.);
        assert_abs_diff_eq!(x[2], 1.);
        assert_abs_diff_eq!(x[3], 1.);
        assert_abs_diff_eq!(x[4], 1.);
    }
}
