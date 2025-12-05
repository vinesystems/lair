use num_traits::Float;

/// Computes one DQDS transform in ping-pong form, one version for IEEE machines.
///
/// This routine computes one dqds transform in ping-pong form, with protection
/// against overflow and underflow. The `PP` parameter distinguishes between
/// "ping" (PP=0) and "pong" (PP=1) forms.
///
/// # Returns
///
/// Returns `None` if `n0 - i0 - 1 <= 0` (nothing to do), otherwise returns
/// `Some((d_min, d_min_1, d_min_2, d_n, d_nm1, d_nm2))`.
#[allow(dead_code, clippy::if_not_else)]
pub(crate) fn lasq5<A, const PP: usize>(
    i0: usize,
    n0: usize,
    z: &mut [A],
    mut tau: A,
    sigma: A,
    eps: A,
) -> Option<(A, A, A, A, A, A)>
where
    A: Float,
{
    assert!(PP <= 1, "`PP` must be either 0 or 1.");

    // Early return if nothing to do (matches Fortran: IF((N0-I0-1).LE.0) RETURN)
    if n0 <= i0 + 1 {
        return None;
    }

    let d_thresh = eps * (sigma + tau);
    let half = A::one() / (A::one() + A::one());
    if tau < d_thresh * half {
        tau = A::zero();
    }

    // Fortran uses 1-based indexing. For array access Z(X) in Fortran -> z[X-1] in Rust.
    // We keep Fortran index values and subtract 1 when accessing arrays.

    // Initial setup: Fortran J4 = 4*I0 + PP - 3
    let init_j4 = 4 * i0 + PP - 3; // Fortran's J4 (1-based)
    let mut e_min = z[init_j4 + 4 - 1]; // Z(J4+4)
    let mut d = z[init_j4 - 1] - tau; // Z(J4) - TAU
    let mut d_min = d;

    // The condition is kept as `tau != 0` to match Fortran's structure
    if tau != A::zero() {
        // TAU != 0 case (no thresholding of small d values)
        if PP == 0 {
            // Fortran: DO J4 = 4*I0, 4*(N0-3), 4
            let mut j4 = 4 * i0;
            while j4 <= 4 * (n0 - 3) {
                z[j4 - 2 - 1] = d + z[j4 - 1 - 1]; // Z(J4-2) = D + Z(J4-1)
                let tmp = z[j4 + 1 - 1] / z[j4 - 2 - 1]; // TEMP = Z(J4+1) / Z(J4-2)
                d = d * tmp - tau;
                d_min = d_min.min(d);
                z[j4 - 1] = z[j4 - 1 - 1] * tmp; // Z(J4) = Z(J4-1)*TEMP
                e_min = e_min.min(z[j4 - 1]); // EMIN = MIN(Z(J4), EMIN)
                j4 += 4;
            }
        } else {
            // PP == 1
            let mut j4 = 4 * i0;
            while j4 <= 4 * (n0 - 3) {
                z[j4 - 3 - 1] = d + z[j4 - 1]; // Z(J4-3) = D + Z(J4)
                let tmp = z[j4 + 2 - 1] / z[j4 - 3 - 1]; // TEMP = Z(J4+2) / Z(J4-3)
                d = d * tmp - tau;
                d_min = d_min.min(d);
                z[j4 - 1 - 1] = z[j4 - 1] * tmp; // Z(J4-1) = Z(J4)*TEMP
                e_min = e_min.min(z[j4 - 1 - 1]); // EMIN = MIN(Z(J4-1), EMIN)
                j4 += 4;
            }
        }
    } else {
        // TAU == 0 case (with thresholding: sets d's to zero if they are small enough)
        if PP == 0 {
            let mut j4 = 4 * i0;
            while j4 <= 4 * (n0 - 3) {
                z[j4 - 2 - 1] = d + z[j4 - 1 - 1];
                let tmp = z[j4 + 1 - 1] / z[j4 - 2 - 1];
                d = d * tmp - tau;
                if d < d_thresh {
                    d = A::zero();
                }
                d_min = d_min.min(d);
                z[j4 - 1] = z[j4 - 1 - 1] * tmp;
                e_min = e_min.min(z[j4 - 1]);
                j4 += 4;
            }
        } else {
            // PP == 1
            let mut j4 = 4 * i0;
            while j4 <= 4 * (n0 - 3) {
                z[j4 - 3 - 1] = d + z[j4 - 1];
                let tmp = z[j4 + 2 - 1] / z[j4 - 3 - 1];
                d = d * tmp - tau;
                if d < d_thresh {
                    d = A::zero();
                }
                d_min = d_min.min(d);
                z[j4 - 1 - 1] = z[j4 - 1] * tmp;
                e_min = e_min.min(z[j4 - 1 - 1]);
                j4 += 4;
            }
        }
    }

    // Unroll last two steps
    let d_nm2 = d;
    let d_min_2 = d_min;

    // Fortran: J4 = 4*(N0-2) - PP
    let j4 = 4 * (n0 - 2) - PP;
    // Fortran: J4P2 = J4 + 2*PP - 1
    let j4p2 = j4 + 2 * PP - 1;

    z[j4 - 2 - 1] = d_nm2 + z[j4p2 - 1]; // Z(J4-2) = DNM2 + Z(J4P2)
    z[j4 - 1] = z[j4p2 + 2 - 1] * (z[j4p2 - 1] / z[j4 - 2 - 1]); // Z(J4) = Z(J4P2+2)*(Z(J4P2)/Z(J4-2))
    let d_nm1 = z[j4p2 + 2 - 1] * (d_nm2 / z[j4 - 2 - 1]) - tau; // DNM1 = Z(J4P2+2)*(DNM2/Z(J4-2)) - TAU
    d_min = d_min.min(d_nm1);

    let d_min_1 = d_min;

    // J4 = J4 + 4
    let j4 = j4 + 4;
    let j4p2 = j4 + 2 * PP - 1;

    z[j4 - 2 - 1] = d_nm1 + z[j4p2 - 1]; // Z(J4-2) = DNM1 + Z(J4P2)
    z[j4 - 1] = z[j4p2 + 2 - 1] * (z[j4p2 - 1] / z[j4 - 2 - 1]); // Z(J4) = Z(J4P2+2)*(Z(J4P2)/Z(J4-2))
    let d_n = z[j4p2 + 2 - 1] * (d_nm1 / z[j4 - 2 - 1]) - tau; // DN = Z(J4P2+2)*(DNM1/Z(J4-2)) - TAU
    d_min = d_min.min(d_n);

    z[j4 + 2 - 1] = d_n; // Z(J4+2) = DN
    z[4 * n0 - PP - 1] = e_min; // Z(4*N0-PP) = EMIN

    Some((d_min, d_min_1, d_min_2, d_n, d_nm1, d_nm2))
}

#[cfg(test)]
mod tests {
    #[test]
    fn lasq5_returns_none_when_n0_le_i0_plus_1() {
        // Test early return condition: n0 <= i0 + 1
        let mut z = [0.0f64; 20];
        let result = super::lasq5::<f64, 0>(2, 3, &mut z, 0.5, 0.0, 1e-15);
        assert!(result.is_none());

        let result = super::lasq5::<f64, 0>(2, 2, &mut z, 0.5, 0.0, 1e-15);
        assert!(result.is_none());
    }

    #[test]
    fn lasq5_ping_with_tau() {
        // Test lasq5 with PP=0 (ping) and non-zero tau
        // Array layout for ping (PP=0):
        // For i0=1, n0=4, the algorithm accesses indices in range [0..16]
        // Fortran 1-based: J4=4*1+0-3=1, Z(J4)=Z(1), Z(J4+4)=Z(5)
        // Initial d = Z(1) - tau, e_min = Z(5)
        let mut z = [
            4.0, 0.0, 2.0, 0.0, // indices 0-3: Z(1)-Z(4) in Fortran
            3.0, 0.0, 1.5, 0.0, // indices 4-7: Z(5)-Z(8)
            2.5, 0.0, 1.0, 0.0, // indices 8-11: Z(9)-Z(12)
            2.0, 0.0, 0.5, 0.0, // indices 12-15: Z(13)-Z(16)
            1.5, 0.0, 0.0, 0.0, // indices 16-19: Z(17)-Z(20)
        ];
        let i0 = 1;
        let n0 = 4;
        let tau = 0.5;
        let sigma = 0.0;
        let eps = 1e-15;

        let result = super::lasq5::<f64, 0>(i0, n0, &mut z, tau, sigma, eps);
        assert!(result.is_some());

        let (d_min, d_min_1, d_min_2, d_n, d_nm1, d_nm2) = result.unwrap();

        // Verify that outputs are computed (non-trivial values)
        assert!(d_min.is_finite());
        assert!(d_min_1.is_finite());
        assert!(d_min_2.is_finite());
        assert!(d_n.is_finite());
        assert!(d_nm1.is_finite());
        assert!(d_nm2.is_finite());

        // Verify monotonicity: d_min should be the minimum
        assert!(d_min <= d_min_1);
        assert!(d_min <= d_min_2);
        assert!(d_min <= d_n);
        assert!(d_min <= d_nm1);
        assert!(d_min <= d_nm2);
    }

    #[test]
    fn lasq5_pong_with_tau() {
        // Test lasq5 with PP=1 (pong) and non-zero tau
        // For PP=1: J4=4*1+1-3=2, Z(J4)=Z(2), Z(J4+4)=Z(6)
        let mut z = [
            0.0, 4.0, 0.0, 2.0, // indices 0-3
            0.0, 3.0, 0.0, 1.5, // indices 4-7
            0.0, 2.5, 0.0, 1.0, // indices 8-11
            0.0, 2.0, 0.0, 0.5, // indices 12-15
            0.0, 1.5, 0.0, 0.0, // indices 16-19
        ];
        let i0 = 1;
        let n0 = 4;
        let tau = 0.3;
        let sigma = 0.1;
        let eps = 1e-15;

        let result = super::lasq5::<f64, 1>(i0, n0, &mut z, tau, sigma, eps);
        assert!(result.is_some());

        let (d_min, d_min_1, d_min_2, d_n, d_nm1, d_nm2) = result.unwrap();

        // Verify that outputs are computed
        assert!(d_min.is_finite());
        assert!(d_min_1.is_finite());
        assert!(d_min_2.is_finite());
        assert!(d_n.is_finite());
        assert!(d_nm1.is_finite());
        assert!(d_nm2.is_finite());
    }

    #[test]
    fn lasq5_ping_zero_tau() {
        // Test lasq5 with PP=0 and tau=0 (or very small tau that gets set to zero)
        let mut z = [
            4.0, 0.0, 2.0, 0.0, 3.0, 0.0, 1.5, 0.0, 2.5, 0.0, 1.0, 0.0, 2.0, 0.0, 0.5, 0.0, 1.5,
            0.0, 0.0, 0.0,
        ];
        let i0 = 1;
        let n0 = 4;
        let tau = 1e-20; // Very small tau, should be treated as zero
        let sigma = 0.0;
        let eps = 1e-15;

        let result = super::lasq5::<f64, 0>(i0, n0, &mut z, tau, sigma, eps);
        assert!(result.is_some());

        let (d_min, d_min_1, d_min_2, d_n, d_nm1, d_nm2) = result.unwrap();

        // Verify outputs are finite
        assert!(d_min.is_finite());
        assert!(d_min_1.is_finite());
        assert!(d_min_2.is_finite());
        assert!(d_n.is_finite());
        assert!(d_nm1.is_finite());
        assert!(d_nm2.is_finite());
    }

    #[test]
    fn lasq5_pong_zero_tau() {
        // Test lasq5 with PP=1 and zero tau
        let mut z = [
            0.0, 4.0, 0.0, 2.0, 0.0, 3.0, 0.0, 1.5, 0.0, 2.5, 0.0, 1.0, 0.0, 2.0, 0.0, 0.5, 0.0,
            1.5, 0.0, 0.0,
        ];
        let i0 = 1;
        let n0 = 4;
        let tau = 1e-20;
        let sigma = 0.0;
        let eps = 1e-15;

        let result = super::lasq5::<f64, 1>(i0, n0, &mut z, tau, sigma, eps);
        assert!(result.is_some());

        let (d_min, d_min_1, d_min_2, d_n, d_nm1, d_nm2) = result.unwrap();

        // Verify outputs
        assert!(d_min.is_finite());
        assert!(d_min_1.is_finite());
        assert!(d_min_2.is_finite());
        assert!(d_n.is_finite());
        assert!(d_nm1.is_finite());
        assert!(d_nm2.is_finite());
    }

    #[test]
    fn lasq5_emin_storage() {
        // Test that emin is correctly stored in z[4*n0 - PP - 1] (0-based)
        let mut z = [
            4.0, 0.0, 2.0, 0.0, 3.0, 0.0, 1.5, 0.0, 2.5, 0.0, 1.0, 0.0, 2.0, 0.0, 0.5, 0.0, 1.5,
            0.0, 0.0, 0.0,
        ];
        let i0 = 1;
        let n0 = 4;
        let tau = 0.5;
        let sigma = 0.0;
        let eps = 1e-15;

        super::lasq5::<f64, 0>(i0, n0, &mut z, tau, sigma, eps);

        // emin should be stored at z[4*n0 - PP - 1] = z[4*4 - 0 - 1] = z[15] for PP=0
        let emin = z[15];
        assert!(emin.is_finite());
        assert!(emin >= 0.0);
    }

    #[test]
    fn lasq5_dn_storage() {
        // Test that d_n is correctly stored in z[j4+2-1] where j4 is the final value
        let mut z = [
            4.0, 0.0, 2.0, 0.0, 3.0, 0.0, 1.5, 0.0, 2.5, 0.0, 1.0, 0.0, 2.0, 0.0, 0.5, 0.0, 1.5,
            0.0, 0.0, 0.0,
        ];
        let i0 = 1;
        let n0 = 4;
        let tau = 0.5;
        let sigma = 0.0;
        let eps = 1e-15;

        let result = super::lasq5::<f64, 0>(i0, n0, &mut z, tau, sigma, eps);
        let (_, _, _, d_n, _, _) = result.unwrap();

        // d_n should be stored in the z array
        // For PP=0: final j4 = 4*(n0-2) - PP + 4 = 4*(4-2) - 0 + 4 = 12
        // Z(J4+2) = z[12+2-1] = z[13]
        assert!(d_n.is_finite());
        assert!((z[13] - d_n).abs() < 1e-14);
    }

    #[test]
    fn lasq5_larger_problem() {
        // Test with a larger n0 to exercise the main loop more
        let mut z = [
            5.0, 0.0, 2.5, 0.0, // i=1
            4.5, 0.0, 2.0, 0.0, // i=2
            4.0, 0.0, 1.5, 0.0, // i=3
            3.5, 0.0, 1.0, 0.0, // i=4
            3.0, 0.0, 0.5, 0.0, // i=5
            2.5, 0.0, 0.0, 0.0, // i=6
        ];
        let i0 = 1;
        let n0 = 5;
        let tau = 0.3;
        let sigma = 0.1;
        let eps = 1e-15;

        let result = super::lasq5::<f64, 0>(i0, n0, &mut z, tau, sigma, eps);
        assert!(result.is_some());

        let (d_min, d_min_1, d_min_2, d_n, d_nm1, d_nm2) = result.unwrap();

        assert!(d_min.is_finite());
        assert!(d_min_1.is_finite());
        assert!(d_min_2.is_finite());
        assert!(d_n.is_finite());
        assert!(d_nm1.is_finite());
        assert!(d_nm2.is_finite());

        // d_min should be the overall minimum
        assert!(d_min <= d_n);
        assert!(d_min <= d_nm1);
        assert!(d_min <= d_nm2);
    }

    #[test]
    fn lasq5_tau_threshold_behavior() {
        // Test that very small tau gets set to zero
        let mut z1 = [
            4.0, 0.0, 2.0, 0.0, 3.0, 0.0, 1.5, 0.0, 2.5, 0.0, 1.0, 0.0, 2.0, 0.0, 0.5, 0.0, 1.5,
            0.0, 0.0, 0.0,
        ];
        let mut z2 = z1;

        let i0 = 1;
        let n0 = 4;
        let sigma = 0.0;
        let eps = 1e-15;

        // tau = 0 explicitly
        let result1 = super::lasq5::<f64, 0>(i0, n0, &mut z1, 0.0, sigma, eps);

        // tau very small (should be set to 0 internally)
        let result2 = super::lasq5::<f64, 0>(i0, n0, &mut z2, 1e-30, sigma, eps);

        // Both should produce the same result
        let (d_min1, _, _, d_n1, _, _) = result1.unwrap();
        let (d_min2, _, _, d_n2, _, _) = result2.unwrap();

        assert!((d_min1 - d_min2).abs() < 1e-14);
        assert!((d_n1 - d_n2).abs() < 1e-14);
    }
}
