use crate::Scalar;
use std::ptr;

#[allow(clippy::cast_possible_wrap)]
pub unsafe fn laswp<A>(
    ncols: usize,
    a: *mut A,
    row_stride: isize,
    col_stride: isize,
    begin: usize,
    piv: &[usize],
) where
    A: Scalar,
{
    for (i, &p) in piv.iter().enumerate().skip(begin) {
        if i == p {
            continue;
        }
        let mut n = ncols;
        let mut row1 = a.offset(i as isize * row_stride);
        let mut row2 = a.offset(p as isize * row_stride);
        loop {
            let mut tmp = ptr::read(row1);
            tmp = ptr::replace(row2, tmp);
            ptr::write(row1, tmp);
            if n == 1 {
                break;
            }
            row1 = row1.offset(col_stride);
            row2 = row2.offset(col_stride);
            n -= 1;
        }
    }
}
