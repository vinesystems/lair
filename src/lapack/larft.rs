use crate::{blas, Scalar};
use ndarray::{s, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
use std::cmp;

/// Forms the triangular factor of a block reflector.
#[allow(dead_code)]
pub fn forward_columnwise<A, SV, ST>(v: &ArrayBase<SV, Ix2>, tau: &ArrayBase<ST, Ix1>) -> Array2<A>
where
    A: Scalar,
    SV: Data<Elem = A>,
    ST: Data<Elem = A>,
{
    let mut triangular = Array2::zeros((v.ncols(), v.ncols()));
    if v.is_empty() {
        return triangular;
    }

    let mut prev_last_v = v.nrows();
    for (i, ((v_row_i, v_col_i), &tau_i)) in (v.rows().into_iter().zip(v.columns().into_iter()))
        .zip(tau.iter())
        .take(v.ncols())
        .enumerate()
    {
        if tau_i == A::zero() {
            continue;
        }

        for (&v_i_j, t_j_i) in v_row_i
            .iter()
            .zip(triangular.column_mut(i).into_iter())
            .take(i)
        {
            *t_j_i = -tau_i * v_i_j;
        }
        prev_last_v = cmp::max(i, prev_last_v);
        let (last_v, _) = v_col_i
            .into_iter()
            .enumerate()
            .rev()
            .take(v.nrows() - i)
            .find(|(_, &elem)| elem != A::zero())
            .unwrap_or((i, &A::zero()));
        let j = cmp::min(last_v, prev_last_v);
        blas::gemv::conjtrans(
            -tau_i,
            &v.slice(s![i + 1..=j, ..i]),
            &v_col_i.slice(s![i + 1..=j]),
            A::one(),
            &mut triangular.column_mut(i),
        );

        let (a, mut x) = triangular.view_mut().split_at(Axis(1), i);
        blas::trmv::upper_notrans(&a, &mut x.column_mut(0).slice_mut(s![..i]));
        triangular[(i, i)] = tau_i;
        if i > 0 {
            prev_last_v = cmp::max(prev_last_v, last_v);
        } else {
            prev_last_v = last_v;
        }
    }
    triangular
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2};

    #[test]
    fn forward_columnwise() {
        let v = arr2(&[[1., 0.], [-1., 1.], [3., 2.]]);
        let tau = arr1(&[-4., 1.]);
        let t = super::forward_columnwise(&v, &tau);
        assert_abs_diff_eq!(t, arr2(&[[-4., 20.], [0., 1.]]));
    }
}
