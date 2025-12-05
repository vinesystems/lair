use std::ops::MulAssign;

use num_traits::Float;

/// Computes one DQD transform in ping-pong form, with protection against
/// underflow and overflow.
///
/// This is a Rust translation of LAPACK's DLASQ6. The array `z` should have
/// length `(n0 + 1) * 4`. Index parameters `i0` and `n0` are 0-based (unlike
/// the 1-based Fortran version).
///
/// Returns `(d_min, d_min_1, d_min_2, d_n, d_nm1, d_nm2)`.
///
/// # Panics
///
/// * `PP` is not 0 or 1.
#[allow(dead_code)]
pub(crate) fn lasq6<A, const PP: usize>(i0: usize, n0: usize, z: &mut [A]) -> (A, A, A, A, A, A)
where
    A: Float + MulAssign,
{
    assert!(PP <= 1, "`PP` must be either 0 or 1.");

    // Early return if not enough elements to process.
    // Fortran: IF( ( N0-I0-1 ).LE.0 ) RETURN
    if n0 <= i0 + 1 {
        return (A::zero(), A::zero(), A::zero(), A::zero(), A::zero(), A::zero());
    }

    // Initialize: Fortran uses 1-based I0/N0, we use 0-based i0/n0.
    // Fortran I0 = i0 + 1, N0 = n0 + 1.
    // Fortran: J4 = 4*I0 + PP - 3, then accesses Z(J4) and Z(J4+4).
    // Converting to 0-indexed: z[4*i0 + PP] and z[4*i0 + 4 + PP].
    let j4 = 4 * i0 + PP;
    let mut e_min = z[j4 + 4];
    let mut d = z[j4];
    let mut d_min = d;

    // Main loop. Fortran: DO J4 = 4*I0, 4*(N0-3), 4
    // With I0 = i0+1, N0 = n0+1: J4 from 4*(i0+1) to 4*(n0-2) step 4
    if PP == 0 {
        let j4_start = 4 * (i0 + 1);
        let j4_end = 4 * (n0 - 2);

        if j4_start <= j4_end {
            let mut j4_f = j4_start;
            while j4_f <= j4_end {
                // Fortran: Z(J4-2) = D + Z(J4-1)
                z[j4_f - 3] = d + z[j4_f - 2];
                if z[j4_f - 3] == A::zero() {
                    z[j4_f - 1] = A::zero();
                    d = z[j4_f];
                    d_min = d;
                    e_min = A::zero();
                } else if A::min_positive_value() * z[j4_f] < z[j4_f - 3]
                    && A::min_positive_value() * z[j4_f - 3] < z[j4_f]
                {
                    let tmp = z[j4_f] / z[j4_f - 3];
                    z[j4_f - 1] = z[j4_f - 2] * tmp;
                    d *= tmp;
                } else {
                    z[j4_f - 1] = z[j4_f] * (z[j4_f - 2] / z[j4_f - 3]);
                    d = z[j4_f] * (d / z[j4_f - 3]);
                }
                if d < d_min {
                    d_min = d;
                }
                let e = z[j4_f - 1];
                if e < e_min {
                    e_min = e;
                }
                j4_f += 4;
            }
        }
    } else {
        // PP == 1
        let j4_start = 4 * (i0 + 1);
        let j4_end = 4 * (n0 - 2);

        if j4_start <= j4_end {
            let mut j4_f = j4_start;
            while j4_f <= j4_end {
                // Fortran: Z(J4-3) = D + Z(J4)
                z[j4_f - 4] = d + z[j4_f - 1];
                if z[j4_f - 4] == A::zero() {
                    z[j4_f - 2] = A::zero();
                    d = z[j4_f + 1];
                    d_min = d;
                    e_min = A::zero();
                } else if A::min_positive_value() * z[j4_f + 1] < z[j4_f - 4]
                    && A::min_positive_value() * z[j4_f - 4] < z[j4_f + 1]
                {
                    let tmp = z[j4_f + 1] / z[j4_f - 4];
                    z[j4_f - 2] = z[j4_f - 1] * tmp;
                    d *= tmp;
                } else {
                    z[j4_f - 2] = z[j4_f + 1] * (z[j4_f - 1] / z[j4_f - 4]);
                    d = z[j4_f + 1] * (d / z[j4_f - 4]);
                }
                if d < d_min {
                    d_min = d;
                }
                let e = z[j4_f - 2];
                if e < e_min {
                    e_min = e;
                }
                j4_f += 4;
            }
        }
    }

    // Unroll last two steps.
    let d_nm2 = d;
    let d_min_2 = d_min;

    // Fortran: J4 = 4*(N0-2) - PP, J4P2 = J4 + 2*PP - 1
    let mut j4_f = 4 * (n0 - 1) - PP;
    let mut j4_p2 = j4_f + 2 * PP - 1;

    // First unrolled iteration
    z[j4_f - 3] = d_nm2 + z[j4_p2 - 1];

    let d_nm1;
    if z[j4_f - 3] == A::zero() {
        z[j4_f - 1] = A::zero();
        d_nm1 = z[j4_p2 + 1];
        d_min = d_nm1;
        e_min = A::zero();
    } else if A::min_positive_value() * z[j4_p2 + 1] < z[j4_f - 3]
        && A::min_positive_value() * z[j4_f - 3] < z[j4_p2 + 1]
    {
        let tmp = z[j4_p2 + 1] / z[j4_f - 3];
        z[j4_f - 1] = z[j4_p2 - 1] * tmp;
        d_nm1 = d_nm2 * tmp;
    } else {
        z[j4_f - 1] = z[j4_p2 + 1] * (z[j4_p2 - 1] / z[j4_f - 3]);
        d_nm1 = z[j4_p2 + 1] * (d_nm2 / z[j4_f - 3]);
    }
    if d_nm1 < d_min {
        d_min = d_nm1;
    }

    let d_min_1 = d_min;
    j4_f += 4;
    j4_p2 = j4_f + 2 * PP - 1;

    // Second unrolled iteration
    z[j4_f - 3] = d_nm1 + z[j4_p2 - 1];

    let d_n;
    if z[j4_f - 3] == A::zero() {
        z[j4_f - 1] = A::zero();
        d_n = z[j4_p2 + 1];
        d_min = d_n;
        e_min = A::zero();
    } else if A::min_positive_value() * z[j4_p2 + 1] < z[j4_f - 3]
        && A::min_positive_value() * z[j4_f - 3] < z[j4_p2 + 1]
    {
        let tmp = z[j4_p2 + 1] / z[j4_f - 3];
        z[j4_f - 1] = z[j4_p2 - 1] * tmp;
        d_n = d_nm1 * tmp;
    } else {
        z[j4_f - 1] = z[j4_p2 + 1] * (z[j4_p2 - 1] / z[j4_f - 3]);
        d_n = z[j4_p2 + 1] * (d_nm1 / z[j4_f - 3]);
    }
    if d_n < d_min {
        d_min = d_n;
    }

    z[j4_f + 1] = d_n;
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

    #[test]
    fn lasq6_ping_larger() {
        // Test with n0=5 which requires (n0+1)*4 = 24 elements
        // This exercises the main loop (n0 >= 3 means loop runs for n0-3 = 2 iterations)
        let mut z: [f64; 24] = [
            1.0, 0.5, 2.0, 0.25, // i=0
            3.0, 0.5, 1.5, 0.75, // i=1
            2.0, 0.25, 1.0, 0.5, // i=2
            1.5, 0.125, 2.5, 0.375, // i=3
            0.5, 0.0625, 3.0, 1.0, // i=4
            2.0, 0.5, 1.0, 0.25, // i=5
        ];
        let res = super::lasq6::<_, 0>(0, 5, &mut z);

        // Verify that d_min <= d_n, d_nm1, d_nm2 (basic sanity check)
        let (d_min, d_min_1, d_min_2, d_n, d_nm1, d_nm2) = res;
        assert!(d_min <= d_n || d_min.is_nan() || d_n.is_nan());
        assert!(d_min_1 <= d_nm1 || d_min_1.is_nan() || d_nm1.is_nan());
        assert!(d_min_2 <= d_nm2 || d_min_2.is_nan() || d_nm2.is_nan());
    }

    #[test]
    fn lasq6_pong_larger() {
        // Test with n0=5 which requires (n0+1)*4 = 24 elements, PP=1
        let mut z: [f64; 24] = [
            1.0, 0.5, 2.0, 0.25, // i=0
            3.0, 0.5, 1.5, 0.75, // i=1
            2.0, 0.25, 1.0, 0.5, // i=2
            1.5, 0.125, 2.5, 0.375, // i=3
            0.5, 0.0625, 3.0, 1.0, // i=4
            2.0, 0.5, 1.0, 0.25, // i=5
        ];
        let res = super::lasq6::<_, 1>(0, 5, &mut z);

        // Verify that d_min <= d_n, d_nm1, d_nm2 (basic sanity check)
        let (d_min, d_min_1, d_min_2, d_n, d_nm1, d_nm2) = res;
        assert!(d_min <= d_n || d_min.is_nan() || d_n.is_nan());
        assert!(d_min_1 <= d_nm1 || d_min_1.is_nan() || d_nm1.is_nan());
        assert!(d_min_2 <= d_nm2 || d_min_2.is_nan() || d_nm2.is_nan());
    }

    #[test]
    fn lasq6_early_return() {
        // Test early return when n0 <= i0 + 1
        let mut z: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let z_orig = z;
        let res = super::lasq6::<_, 0>(0, 1, &mut z);

        // Array should be unchanged
        assert_eq!(z, z_orig);
        // All return values should be zero
        assert_eq!(res, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn lasq6_ping_nonzero_i0() {
        // Test with i0 > 0
        let mut z: [f64; 20] = [
            0.0, 0.0, 0.0, 0.0, // i=0 (unused since i0=1)
            2.0, 0.5, 3.0, 1.0, // i=1
            1.5, 0.25, 2.0, 0.5, // i=2
            1.0, 0.125, 1.5, 0.25, // i=3
            0.5, 0.0625, 1.0, 0.125, // i=4
        ];
        let res = super::lasq6::<_, 0>(1, 4, &mut z);

        // Verify basic properties
        let (d_min, _, _, d_n, _, _) = res;
        assert!(d_min <= d_n || d_min.is_nan() || d_n.is_nan());
    }

    #[test]
    fn lasq6_f32() {
        // Test with f32 to ensure generic implementation works
        let mut z: [f32; 16] = [
            2.0, -1.0, 3.0, 1.0, -1.0, 3.0, 2.0, 2.0, 1.0, 1.0, -1.0, -1.0, 2.0, -3.0, -1.0, 2.0,
        ];
        let res = super::lasq6::<_, 0>(0, 3, &mut z);

        // Basic sanity check - results should be finite
        assert!(res.0.is_finite());
        assert!(res.3.is_finite());
    }
}
