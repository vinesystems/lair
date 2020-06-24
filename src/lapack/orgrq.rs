use crate::{blas, lapack, Scalar};
use ndarray::{s, Array2, ArrayBase, Data, Ix1, Ix2};

#[allow(dead_code)]
pub fn orgrq<A, S>(a: &ArrayBase<S, Ix2>, tau: &ArrayBase<S, Ix1>) -> Array2<A>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    let mut q = a.to_owned();
    if tau.len() < a.nrows() {
        for j in 0..q.ncols() {
            for l in 0..q.nrows() - tau.len() {
                q[(l, j)] = A::zero();
            }
            if j >= q.ncols() - q.nrows() && j < q.ncols() - tau.len() {
                q[(j + a.nrows() - a.ncols(), j)] = A::one();
            }
        }
    }
    for i in 0..tau.len() {
        let row = a.nrows() - tau.len() + i;
        q[(row, row + a.ncols() - a.nrows())] = A::one();
        let v = q
            .row(row)
            .slice(s![..=row + a.ncols() - a.nrows()])
            .to_owned();
        lapack::larf::right(
            &v,
            tau[i],
            &mut q.slice_mut(s![..row, ..=row + a.ncols() - a.nrows()]),
        );
        blas::scal(
            -tau[i],
            &mut q.row_mut(row).slice_mut(s![..row + a.ncols() - a.nrows()]),
        );
        q[(row, row + a.ncols() - a.nrows())] = A::one() - tau[i];

        for j in row + 1 + a.ncols() - a.nrows()..a.ncols() {
            q[(row, j)] = A::zero();
        }
    }
    q
}
