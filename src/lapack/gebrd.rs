use std::ops::{Div, MulAssign};

use ndarray::{s, Array1, ArrayBase, Axis, DataMut, Ix2};

use crate::{lapack, Scalar};

/// An upper or lower bidiagonal matrix.
///
/// The bidiagonal matrix B is Q^H * `a` * P, and consists of four components:
///
/// 1. The diagonal elements
/// 2. The off-diagonal elements
/// 3. The scalar factors of the elementary reflectors representing Q
/// 4. The scalar factors of the elementary reflectors representing P
pub type Bidiagonal<A> = (
    Array1<<A as Scalar>::Real>,
    Array1<<A as Scalar>::Real>,
    Array1<A>,
    Array1<A>,
);

#[allow(dead_code)]
pub fn tall<A, S>(a: &mut ArrayBase<S, Ix2>) -> Bidiagonal<A>
where
    A: Scalar + Div<<A as Scalar>::Real, Output = A> + MulAssign<<A as Scalar>::Real>,
    S: DataMut<Elem = A>,
{
    unblocked_tall(a)
}

fn unblocked_tall<A, S>(a: &mut ArrayBase<S, Ix2>) -> Bidiagonal<A>
where
    A: Scalar + Div<<A as Scalar>::Real, Output = A> + MulAssign<<A as Scalar>::Real>,
    S: DataMut<Elem = A>,
{
    assert!(a.nrows() >= a.ncols());
    if a.is_empty() {
        return (
            Array1::<A::Real>::zeros(0),
            Array1::<A::Real>::zeros(0),
            Array1::<A>::zeros(0),
            Array1::<A>::zeros(0),
        );
    }

    let (mut d, mut e, mut tau_q, mut tau_p) = unsafe {
        // Will be initialized in the following `for` loop.
        (
            Array1::<A::Real>::uninit(a.ncols()).assume_init(),
            Array1::<A::Real>::uninit(a.ncols() - 1).assume_init(),
            Array1::<A>::uninit(a.ncols()).assume_init(),
            Array1::<A>::uninit(a.ncols()).assume_init(),
        )
    };
    for (i, (d_i, (q_i, p_i))) in d
        .iter_mut()
        .zip(tau_q.iter_mut().zip(tau_p.iter_mut()))
        .enumerate()
    {
        let mut a_ii = a.slice_mut(s![i.., i..]);
        let alpha = *a_ii.first().expect("nonempty");
        let (beta, _, tau) = lapack::larfg(alpha, a_ii.column_mut(0).slice_mut(s![1..]));
        *d_i = beta;
        *q_i = tau;
        *a_ii.first_mut().expect("nonempty") = A::one();

        let (mut left, mut right) = a_ii.split_at(Axis(1), 1);
        lapack::larf::left(&left.column(0), tau.conj(), &mut right);
        *left.first_mut().expect("nonempty") = beta.into();
        if i < a.ncols() - 1 {
            lapack::lacgv(&mut a.row_mut(i).slice_mut(s![i + 1..]));
            let alpha = a[(i, i + 1)];
            let (beta, _, tau) = lapack::larfg(alpha, a.row_mut(i).slice_mut(s![i + 2..]));
            e[i] = beta;
            *p_i = tau;
            a[(i, i + 1)] = A::one();

            let (mut upper, mut lower) = a.slice_mut(s![i.., i + 1..]).split_at(Axis(0), 1);
            lapack::larf::right(&upper.row(0), *p_i, &mut lower);
            lapack::lacgv(&mut upper.row_mut(0));
            *upper.first_mut().expect("nonempty") = e[i].into();
        } else {
            *p_i = A::zero();
        }
    }
    (d, e, tau_q, tau_p)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2};

    #[test]
    fn tall() {
        let mut a = arr2(&[
            [1_f64, 2_f64, 3_f64, 1_f64],
            [2_f64, 2_f64, 1_f64, 3_f64],
            [3_f64, 1_f64, 2_f64, 2_f64],
            [2_f64, 3_f64, 3_f64, 1_f64],
            [1_f64, 3_f64, 1_f64, 3_f64],
        ]);
        let (d, e, tau_q, tau_p) = super::tall(&mut a);
        assert_abs_diff_eq!(
            a,
            arr2(&[
                [
                    -4.358898943540673,
                    7.152474728151237,
                    0.36602540378443854,
                    0.36602540378443865
                ],
                [
                    0.37321099372674155,
                    -3.7204979859096694,
                    1.1403421539769902,
                    0.41421356237309526
                ],
                [
                    0.5598164905901123,
                    0.7245458504029912,
                    -0.836432765895214,
                    0.0000000000000022881872238998206
                ],
                [
                    0.37321099372674155,
                    -0.018780759811874194,
                    0.9396262080188605,
                    2.8284271247461903
                ],
                [
                    0.18660549686337077,
                    -0.4883921008919025,
                    0.30675112209074024,
                    -0.9081340845417558
                ]
            ]),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            d,
            arr1(&[
                -4.358898943540673,
                -3.7204979859096694,
                -0.836432765895214,
                2.8284271247461903,
            ]),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            e,
            arr1(&[
                7.152474728151237,
                1.1403421539769902,
                0.0000000000000022881872238998206,
            ]),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            tau_q,
            arr1(&[
                1.2294157338705618,
                1.133885677079638,
                1.0116370318963936,
                1.0960660725096263,
            ]),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            tau_p,
            arr1(&[1.577350269189626, 1.7071067811865475, 0.0, 0.0,]),
            epsilon = 1e-12
        );
    }
}
