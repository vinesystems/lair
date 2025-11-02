use std::ops::MulAssign;

use num_traits::Float;

/// Computes one DQD transform in ping-pong form, with protection against
/// underflow and overflow.
///
/// # Panics
///
/// * The number of elements in `z` is not a multiple of 4.
/// * The length of `z` is not `(n0 + 1) * 4`.
/// * `n0` is not greater than `i0 + 1`.
#[allow(dead_code)]
pub(crate) fn lasq6<A, const PP: usize>(i0: usize, n0: usize, z: &mut [A]) -> (A, A, A, A, A, A)
where
    A: Float + MulAssign,
{
    assert!(PP <= 1, "`PP` must be either 0 or 1.");
    assert_eq!(z.len(), (n0 + 1) * 4);
    assert!(i0 + 1 < n0);

    let mut j4 = 4 * i0 + PP;
    let mut e_min = z[j4 + 4];
    let mut d = z[j4];
    let mut d_min = d;

    if n0 >= 3 {
        for j4 in (4 * i0 + 3..=4 * (n0 - 3) + 3).step_by(4) {
            z[j4 - PP - 2] = d + z[j4 + PP - 1];
            if z[j4 - PP - 2] == A::zero() {
                z[j4 - PP] = A::zero();
                d = z[j4 + PP + 1];
                d_min = d;
                e_min = A::zero();
            } else if A::min_positive_value() * z[j4 + PP + 1] < z[j4 - PP - 2]
                && A::min_positive_value() * z[j4 - PP - 2] < z[j4 + PP + 1]
            {
                let tmp = z[j4 + PP + 1] / z[j4 - PP - 2];
                z[j4 - PP] = z[j4 + PP - 1] * tmp;
                d *= tmp;
            } else {
                z[j4 - PP] = z[j4 + PP + 1] * (z[j4 + PP - 1] / z[j4 - PP - 2]);
                d = z[j4 + PP + 1] * (d / z[j4 - PP - 2]);
            }
            if d < d_min {
                d_min = d;
            }
            let e = z[j4 - PP];
            if e < e_min {
                e_min = e;
            }
        }
    }

    let d_nm2 = d;
    let d_min_2 = d_min;
    j4 = 4 * (n0 - 2) + 3 - PP;
    let mut j4_p2 = j4 + 2 * PP - 1;
    z[j4 - 2] = d_nm2 + z[j4_p2];
    let d_nm1;
    if z[j4 - 2] == A::zero() {
        z[j4] = A::zero();
        d_nm1 = z[j4_p2 + 2];
        d_min = d_nm1;
        e_min = A::zero();
    } else if A::min_positive_value() * z[j4_p2 + 2] < z[j4 - 2]
        && A::min_positive_value() * z[j4 - 2] < z[j4_p2 + 2]
    {
        let tmp = z[j4_p2 + 2] / z[j4 - 2];
        z[j4] = z[j4_p2] * tmp;
        d_nm1 = d_nm2 * tmp;
    } else {
        z[j4] = z[j4_p2 + 2] * (z[j4_p2] / z[j4 - 2]);
        d_nm1 = z[j4_p2 + 2] * (d_nm2 / z[j4 - 2]);
    }
    if d_nm1 < d_min {
        d_min = d_nm1;
    }

    let d_min_1 = d_min;
    j4 += 4;
    j4_p2 = j4 + 2 * PP - 1;
    z[j4 - 2] = d_nm1 + z[j4_p2];
    let d_n;
    if z[j4 - 2] == A::zero() {
        z[j4] = A::zero();
        d_n = z[j4_p2 + 2];
        d_min = d_n;
        e_min = A::zero();
    } else if A::min_positive_value() * z[j4_p2 + 2] < z[j4 - 2]
        && A::min_positive_value() * z[j4 - 2] < z[j4_p2 + 2]
    {
        let tmp = z[j4_p2 + 2] / z[j4 - 2];
        z[j4] = z[j4_p2] * tmp;
        d_n = d_nm1 * tmp;
    } else {
        z[j4] = z[j4_p2 + 2] * (z[j4_p2] / z[j4 - 2]);
        d_n = z[j4_p2 + 2] * (d_nm1 / z[j4 - 2]);
    }
    if d_n < d_min {
        d_min = d_n;
    }

    z[j4 + 2] = d_n;
    z[4 * n0 + 3 - PP] = e_min;
    (d_min, d_min_1, d_min_2, d_n, d_nm1, d_nm2)
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;

    #[test]
    fn lasq6_ping() {
        let mut z = [
            2.0, -1.0, 3.0, 1.0, -1.0, 3.0, 2.0, 2.0, 1.0, 1.0, -1.0, -1.0, 2.0, -3.0, -1.0, 2.0,
        ];
        let res = super::lasq6::<_, 0>(0, 3, &mut z);
        assert_eq!(
            z,
            [
                2.0, 5.0, 3.0, -0.6, -1.0, 1.6, 2.0, 1.25, 1.0, -1.25, -1.0, 1.6, 2.0, 0.4, -1.0,
                -1.0,
            ]
        );
        assert_eq!(res, (-0.4, -0.4, -0.4, 0.4, -0.25, -0.4));
    }

    #[test]
    fn lasq6_pong() {
        let mut z = [
            2.0, -1.0, 3.0, 1.0, -1.0, 3.0, 2.0, 2.0, 1.0, 1.0, -1.0, -1.0, 2.0, -3.0, -1.0, 2.0,
        ];
        let z_after = [
            0.0, -1.0, 0.0, 1.0, 5.0, 3.0, 0.4, 2.0, -0.4, 1.0, -7.5, -1.0, 4.5, -3.0, 0.0, 2.0,
        ];
        let res = super::lasq6::<_, 1>(0, 3, &mut z);
        for (&actual, &expected) in z.iter().zip(z_after.iter()) {
            assert_ulps_eq!(actual, expected);
        }
        assert_ulps_eq!(res.0, 0.6);
        assert_ulps_eq!(res.1, 0.6);
        assert_ulps_eq!(res.2, 3.0);
        assert_ulps_eq!(res.3, 4.5);
        assert_ulps_eq!(res.4, 0.6);
        assert_ulps_eq!(res.5, 3.0);
    }
}
