use num_traits::{Float, Zero};

use crate::Scalar;

#[allow(clippy::cast_possible_wrap)]
pub(crate) unsafe fn iamax<A>(n: usize, x: *const A, incx: isize) -> (usize, A::Real)
where
    A: Scalar,
{
    let mut max_val = A::Real::zero();
    let mut max_idx = 0;
    for i in 0..n {
        let elem = x.offset(incx * i as isize);
        let val = (*elem).re().abs() + (*elem).im().abs();
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    (max_idx, max_val)
}

#[cfg(test)]
mod tests {
    #[test]
    fn iamax() {
        let (idx, max) = unsafe {
            let x = [1., 3., 2.];
            super::iamax(x.len(), &x as *const f64, 1)
        };
        assert_eq!(idx, 1);
        assert_eq!(max, 3.);
    }
}
