use crate::Scalar;
use ndarray::ArrayViewMut2;

#[allow(clippy::cast_possible_wrap)]
pub unsafe fn trsm<A>(a: *const A, row_stride: isize, col_stride: isize, mut b: ArrayViewMut2<A>)
where
    A: Scalar,
{
    for j in 0..b.ncols() {
        for k in 0..b.nrows() {
            if *b.uget((k, j)) == A::zero() {
                continue;
            }
            for i in k + 1..b.nrows() {
                let prod =
                    *b.uget((k, j)) * *a.offset(row_stride * i as isize + col_stride * k as isize);
                *b.uget_mut((i, j)) -= prod;
            }
        }
    }
}
