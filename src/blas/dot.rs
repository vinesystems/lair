use std::iter::Sum;
use std::ops::Mul;

#[allow(clippy::cast_possible_wrap, clippy::similar_names)]
pub unsafe fn dot<T>(n: usize, x: *const T, incx: isize, y: *const T, incy: isize) -> T
where
    T: Copy + Mul + Sum<<T as Mul>::Output>,
{
    (0..n)
        .map(|i| *x.offset(incx * i as isize) * *y.offset(incy * i as isize))
        .sum()
}
