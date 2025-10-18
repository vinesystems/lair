use std::ops::MulAssign;

use ndarray::{ArrayBase, Data, DataMut, Ix2};

use crate::Scalar;

/// Computes `b` * `a` assuming `a` is a lower triangular matrix.
///
/// `UNIT` indicates whether to assume `a`'s diagonal elements are 1s.
///
/// # Panics
///
/// Panics if `a` is not a square matrix, or `a`'s # of rows is different from
/// `b`'s # of columns.
pub fn right_lower_notrans<A, SA, SB, const UNIT: bool>(
    a: &ArrayBase<SA, Ix2>,
    b: &mut ArrayBase<SB, Ix2>,
) where
    A: Scalar,
    SA: Data<Elem = A>,
    SB: DataMut<Elem = A>,
{
    assert!(a.is_square());
    assert_eq!(b.ncols(), a.nrows());
    for (j, a_col) in a.columns().into_iter().enumerate() {
        if !UNIT {
            b.column_mut(j).mul_assign(a_col[j]);
        }
        for k in j + 1..b.ncols() {
            let multiplier = a_col[k];
            if multiplier == A::zero() {
                continue;
            }

            for i in 0..b.nrows() {
                let b_i_k = b[(i, k)];
                b[(i, j)] += multiplier * b_i_k;
            }
        }
    }
}

/// Computes `b` * `a^H` assuming `a` is a upper triangular matrix.
///
/// `UNIT` indicates whether to assume `a`'s diagonal elements are 1s.
///
/// # Panics
///
/// Panics if `a` is not a square matrix, or `a`'s # of rows is different from
/// `b`'s # of columns.
pub fn right_upper_conjtrans<A, SA, SB, const UNIT: bool>(
    a: &ArrayBase<SA, Ix2>,
    b: &mut ArrayBase<SB, Ix2>,
) where
    A: Scalar,
    SA: Data<Elem = A>,
    SB: DataMut<Elem = A>,
{
    assert!(a.is_square());
    assert_eq!(b.ncols(), a.nrows());
    for k in 0..a.ncols() {
        for j in 0..k {
            let multiplier = a[(j, k)].conj();
            if multiplier == A::zero() {
                continue;
            }

            for i in 0..b.nrows() {
                let b_i_k = b[(i, k)];
                b[(i, j)] += multiplier * b_i_k;
            }
        }
        if !UNIT {
            let multiplier = a[(k, k)].conj();
            if multiplier != A::one() {
                b.column_mut(k).mul_assign(multiplier);
            }
        }
    }
}

/// Computes `b` * `a^H` assuming `a` is a lower triangular matrix.
///
/// `UNIT` indicates whether to assume `a`'s diagonal elements are 1s.
///
/// # Panics
///
/// Panics if `a` is not a square matrix, or `a`'s # of rows is different from
/// `b`'s # of columns.
#[allow(dead_code)]
pub fn right_lower_conjtrans<A, SA, SB, const UNIT: bool>(
    a: &ArrayBase<SA, Ix2>,
    b: &mut ArrayBase<SB, Ix2>,
) where
    A: Scalar,
    SA: Data<Elem = A>,
    SB: DataMut<Elem = A>,
{
    assert!(a.is_square());
    assert_eq!(b.ncols(), a.nrows());
    for k in (0..a.ncols()).rev() {
        for j in k + 1..b.ncols() {
            let multiplier = a[(j, k)].conj();
            if multiplier == A::zero() {
                continue;
            }

            for i in 0..b.nrows() {
                let increase = multiplier * b[(i, k)];
                b[(i, j)] += increase;
            }
        }
        if !UNIT {
            let multiplier = a[(k, k)].conj();
            if multiplier != A::one() {
                b.column_mut(k).mul_assign(multiplier);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;
    use num_complex::Complex64;

    #[test]
    fn right_lower_notrans_nounit() {
        let a = arr2(&[[-1., 2.], [1., -3.]]);
        let mut b = arr2(&[[-2., 3.], [-3., 1.]]);
        super::right_lower_notrans::<_, _, _, false>(&a, &mut b);
        assert_eq!(b, arr2(&[[5., -9.], [4., -3.]]));
    }

    #[test]
    fn right_upper_conjtrans_nounit() {
        let a = arr2(&[
            [Complex64::new(-1., 2.), Complex64::new(2., -3.)],
            [Complex64::new(2., 1.), Complex64::new(-3., 1.)],
        ]);
        let mut b = arr2(&[
            [Complex64::new(-2., 1.), Complex64::new(3., 1.)],
            [Complex64::new(-3., -1.), Complex64::new(1., 2.)],
        ]);
        super::right_upper_conjtrans::<_, _, _, false>(&a, &mut b);
        assert_eq!(
            b,
            arr2(&[
                [Complex64::new(7., 14.), Complex64::new(-8., -6.)],
                [Complex64::new(-3., 14.), Complex64::new(-1., -7.)]
            ])
        );
    }

    #[test]
    fn right_lower_conjtrans_nounit() {
        let a = arr2(&[
            [Complex64::new(-1., 2.), Complex64::new(2., -3.)],
            [Complex64::new(2., 1.), Complex64::new(-3., 1.)],
        ]);
        let mut b = arr2(&[
            [Complex64::new(-2., 1.), Complex64::new(3., 1.)],
            [Complex64::new(-3., -1.), Complex64::new(1., 2.)],
        ]);
        super::right_lower_conjtrans::<_, _, _, false>(&a, &mut b);
        assert_eq!(
            b,
            arr2(&[
                [Complex64::new(4., 3.), Complex64::new(-11., -2.)],
                [Complex64::new(1., 7.), Complex64::new(-8., -6.)]
            ])
        );
    }
}
