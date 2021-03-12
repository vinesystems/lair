use crate::{Float, One, Real, Scalar, Zero};
use ndarray::{ArrayBase, DataMut, Ix2};

/// Multiplies a full matrix by a real scalar `c_to / c_from`.
#[allow(dead_code)]
pub(crate) fn full<A, S>(c_from: A::Real, c_to: A::Real, a: &mut ArrayBase<S, Ix2>)
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    let small_num = A::Real::sfmin();
    let large_num = A::Real::one() / small_num;

    let mut c_from = c_from;
    let mut c_to = c_to;

    loop {
        let (mul, done) = {
            let small_from = c_from * small_num;
            if small_from == c_from {
                ((c_to / c_from).into(), true)
            } else {
                let small_to = c_to / large_num;
                if small_to == c_to {
                    c_from = A::Real::one();
                    ((c_to).into(), true)
                } else if small_from.abs() > c_to.abs() && c_to.abs() != A::Real::zero() {
                    c_from = small_from;
                    ((small_num).into(), false)
                } else if small_to.abs() > c_from {
                    c_to = small_to;
                    ((large_num).into(), false)
                } else {
                    ((c_to / c_from).into(), true)
                }
            }
        };

        for v in a.iter_mut() {
            *v *= mul;
        }

        if done {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    #[test]
    fn full() {
        let mut a = arr2(&[[3., 2.], [4., 3.]]);
        super::full(2., 4., &mut a);
        assert_eq!(a, arr2(&[[6., 4.], [8., 6.]]));
    }
}
