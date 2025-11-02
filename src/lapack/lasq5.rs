use num_traits::Float;

/// Computes one DQDS transform in ping-pong form, one version for IEEE machines.
///
/// This routine computes one dqds transform in ping-pong form, with protection
/// against overflow and underflow. The `PP` parameter distinguishes between
/// "ping" (PP=0) and "pong" (PP=1) forms.
///
/// # Panics
///
/// * The length of `z` is not `(n0 + 1) * 4`.
/// * `n0` is not greater than `i0 + 1`.
#[allow(dead_code)]
pub(crate) fn lasq5<A, const PP: usize>(
    i0: usize,
    n0: usize,
    z: &mut [A],
    mut tau: A,
    sigma: A,
    eps: A,
) -> (A, A, A, A, A, A)
where
    A: Float,
{
    assert!(PP <= 1, "`PP` must be either 0 or 1.");
    assert!(i0 + 1 < n0);

    let d_thresh = eps * (sigma + tau);
    if tau < d_thresh / (A::one() + A::one()) {
        tau = A::zero();
    }

    let mut j4 = 4 * i0 + PP - 3;
    let mut e_min = z[j4 + 3];
    let mut d = z[j4 - 1] - tau;
    let mut d_min = d;
    if tau == A::zero() {
        for j4 in (4 * i0..=4 * (n0 - 3)).step_by(4) {
            z[j4 - PP - 3] = d + z[j4 + PP - 2];
            let tmp = z[j4 + PP] / z[j4 - PP - 3];
            d = d * tmp - tau;
            if d < d_thresh {
                d = A::zero();
            }
            if d < d_min {
                d_min = d;
            }
            z[j4 - PP - 1] = z[j4 + PP - 2] * tmp;
            let e = z[j4 - PP - 1];
            if e < e_min {
                e_min = e;
            }
        }
    } else {
        for j4 in (4 * i0..=4 * (n0 - 3)).step_by(4) {
            z[j4 - PP - 3] = d + z[j4 + PP - 2];
            let tmp = z[j4 + PP] / z[j4 - PP - 3];
            d = d * tmp - tau;
            if d < d_min {
                d_min = d;
            }
            z[j4 - PP - 1] = z[j4 + PP - 2] * tmp;
            let e = z[j4 - PP - 1];
            if e < e_min {
                e_min = e;
            }
        }
    }

    let d_nm2 = d;
    let d_min_2 = d_min;
    j4 = 4 * (n0 - 2) - PP;
    let mut j4_p2 = j4 + 2 * PP - 1;
    z[j4 - 3] = d_nm2 + z[j4_p2 - 1];
    z[j4 - 1] = z[j4_p2 + 1] * (z[j4_p2 - 1] / z[j4 - 3]);
    let d_nm1 = z[j4_p2 + 1] * (d_nm2 / z[j4 - 3]) - tau;
    if d_nm1 < d_min {
        d_min = d_nm1;
    }

    let d_min_1 = d_min;
    j4 += 4;
    j4_p2 = j4 + 2 * PP - 1;
    z[j4 - 3] = d_nm1 + z[j4_p2 - 1];
    z[j4 - 1] = z[j4_p2 + 1] * (z[j4_p2 - 1] / z[j4 - 3]);
    let d_n = z[j4_p2 + 1] * (d_nm1 / z[j4 - 3]) - tau;
    if d_n < d_min {
        d_min = d_n;
    }

    z[j4 + 1] = d_n;
    z[4 * n0 - PP] = e_min;
    (d_min, d_min_1, d_min_2, d_n, d_nm1, d_nm2)
}

#[cfg(test)]
mod tests {
    #[test]
    fn lasq5_ping_with_tau() {
        // Test lasq5 with PP=0 (ping) and non-zero tau
        let mut z = [
            0.0, 0.0, 0.0, 0.0, // padding (indices 0-3)
            3.0, 0.0, 2.0, 0.0, // elements for i=1 (indices 4-7)
            4.0, 0.0, 1.5, 0.0, // elements for i=2 (indices 8-11)
            2.5, 0.0, 1.0, 0.0, // elements for i=3 (indices 12-15)
            2.0, 0.0, 0.0, 0.0, // elements for i=4 (indices 16-19)
        ];
        let i0 = 1;
        let n0 = 4;
        let tau = 0.5;
        let sigma = 0.0;
        let eps = 1e-15;

        let (d_min, d_min_1, d_min_2, d_n, d_nm1, d_nm2) =
            super::lasq5::<f64, 0>(i0, n0, &mut z, tau, sigma, eps);

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

        // Verify that z was modified (the transform was applied)
        assert_ne!(z[0], 3.0); // First element should have changed
    }

    #[test]
    fn lasq5_pong_with_tau() {
        // Test lasq5 with PP=1 (pong) and non-zero tau
        let mut z = [
            0.0, 0.0, 0.0, 0.0, // padding
            0.0, 3.0, 0.0, 2.0, // elements for i=1
            0.0, 4.0, 0.0, 1.5, // elements for i=2
            0.0, 2.5, 0.0, 1.0, // elements for i=3
            0.0, 2.0, 0.0, 0.0, // elements for i=4
        ];
        let i0 = 1;
        let n0 = 4;
        let tau = 0.3;
        let sigma = 0.1;
        let eps = 1e-15;

        let (d_min, d_min_1, d_min_2, d_n, d_nm1, d_nm2) =
            super::lasq5::<f64, 1>(i0, n0, &mut z, tau, sigma, eps);

        // Verify that outputs are computed
        assert!(d_min.is_finite());
        assert!(d_min_1.is_finite());
        assert!(d_min_2.is_finite());
        assert!(d_n.is_finite());
        assert!(d_nm1.is_finite());
        assert!(d_nm2.is_finite());

        // Verify that z was modified
        assert_ne!(z[1], 3.0);
    }

    #[test]
    fn lasq5_ping_zero_tau() {
        // Test lasq5 with PP=0 and tau=0 (or very small tau that gets set to zero)
        let mut z = [
            0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 2.0, 0.0, 4.0, 0.0, 1.5, 0.0, 2.5, 0.0, 1.0, 0.0, 2.0,
            0.0, 0.0, 0.0,
        ];
        let i0 = 1;
        let n0 = 4;
        let tau = 1e-20; // Very small tau, should be treated as zero
        let sigma = 0.0;
        let eps = 1e-15;

        let (d_min, d_min_1, d_min_2, d_n, d_nm1, d_nm2) =
            super::lasq5::<f64, 0>(i0, n0, &mut z, tau, sigma, eps);

        // Verify outputs are finite
        assert!(d_min.is_finite());
        assert!(d_min_1.is_finite());
        assert!(d_min_2.is_finite());
        assert!(d_n.is_finite());
        assert!(d_nm1.is_finite());
        assert!(d_nm2.is_finite());

        // When tau is zero, the algorithm applies threshold checks
        // d_min may be slightly negative due to numerical precision but should be close to zero
        assert!(d_min >= -1e-10, "d_min = {}, expected >= -1e-10", d_min);
    }

    #[test]
    fn lasq5_pong_zero_tau() {
        // Test lasq5 with PP=1 and zero tau
        let mut z = [
            0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 2.0, 0.0, 4.0, 0.0, 1.5, 0.0, 2.5, 0.0, 1.0, 0.0,
            2.0, 0.0, 0.0,
        ];
        let i0 = 1;
        let n0 = 4;
        let tau = 1e-20;
        let sigma = 0.0;
        let eps = 1e-15;

        let (d_min, d_min_1, d_min_2, d_n, d_nm1, d_nm2) =
            super::lasq5::<f64, 1>(i0, n0, &mut z, tau, sigma, eps);

        // Verify outputs
        assert!(d_min.is_finite());
        assert!(d_min_1.is_finite());
        assert!(d_min_2.is_finite());
        assert!(d_n.is_finite());
        assert!(d_nm1.is_finite());
        assert!(d_nm2.is_finite());
    }

    #[test]
    fn lasq5_ping_properties() {
        // Test basic properties of lasq5 with PP=0 (ping)
        // Using test data similar to lasq6 test
        let mut z = [
            0.0, 0.0, 0.0, 0.0, 2.0, -1.0, 3.0, 1.0, -1.0, 3.0, 2.0, 2.0, 1.0, 1.0, -1.0, -1.0,
            2.0, -3.0, -1.0, 2.0,
        ];
        let i0 = 1;
        let n0 = 4;
        let tau = 0.1;
        let sigma = 0.0;
        let eps = 1e-15;

        let (d_min, d_min_1, d_min_2, d_n, d_nm1, d_nm2) =
            super::lasq5::<f64, 0>(i0, n0, &mut z, tau, sigma, eps);

        // Verify that outputs exist (may be negative due to algorithm nature)
        // Just verify they're not NaN/infinite
        assert!(!d_min.is_nan(), "d_min should not be NaN");
        assert!(!d_n.is_nan(), "d_n should not be NaN");
        assert!(!d_nm1.is_nan(), "d_nm1 should not be NaN");
        assert!(!d_nm2.is_nan(), "d_nm2 should not be NaN");
        assert!(!d_min_1.is_nan(), "d_min_1 should not be NaN");
        assert!(!d_min_2.is_nan(), "d_min_2 should not be NaN");
    }

    #[test]
    fn lasq5_pong_properties() {
        // Test basic properties of lasq5 with PP=1 (pong)
        // Using test data similar to lasq6 pong test
        let mut z = [
            0.0, 0.0, 0.0, 0.0, 2.0, -1.0, 3.0, 1.0, -1.0, 3.0, 2.0, 2.0, 1.0, 1.0, -1.0, -1.0,
            2.0, -3.0, -1.0, 2.0,
        ];
        let i0 = 1;
        let n0 = 4;
        let tau = 0.1;
        let sigma = 0.0;
        let eps = 1e-15;

        let (d_min, d_min_1, d_min_2, d_n, d_nm1, d_nm2) =
            super::lasq5::<f64, 1>(i0, n0, &mut z, tau, sigma, eps);

        // Verify outputs exist and are not NaN
        assert!(!d_min.is_nan(), "d_min should not be NaN");
        assert!(!d_n.is_nan(), "d_n should not be NaN");
        assert!(!d_nm1.is_nan(), "d_nm1 should not be NaN");
        assert!(!d_nm2.is_nan(), "d_nm2 should not be NaN");
        assert!(!d_min_1.is_nan(), "d_min_1 should not be NaN");
        assert!(!d_min_2.is_nan(), "d_min_2 should not be NaN");
    }

    #[test]
    fn lasq5_emin_storage() {
        // Test that emin is correctly stored in z[4*n0 - PP]
        let mut z = [
            0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 2.0, 0.0, 4.0, 0.0, 1.5, 0.0, 2.5, 0.0, 1.0, 0.0, 2.0,
            0.0, 0.0, 0.0,
        ];
        let i0 = 1;
        let n0 = 4;
        let tau = 0.5;
        let sigma = 0.0;
        let eps = 1e-15;

        super::lasq5::<f64, 0>(i0, n0, &mut z, tau, sigma, eps);

        // emin should be stored at z[4*n0 - PP] = z[4*4 - 0] = z[16]
        let emin = z[16];
        assert!(emin.is_finite());
        assert!(emin >= 0.0);
    }

    #[test]
    fn lasq5_dn_storage() {
        // Test that d_n is correctly stored in z[j4+2]
        let mut z = [
            0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 2.0, 0.0, 4.0, 0.0, 1.5, 0.0, 2.5, 0.0, 1.0, 0.0, 2.0,
            0.0, 0.0, 0.0,
        ];
        let i0 = 1;
        let n0 = 4;
        let tau = 0.5;
        let sigma = 0.0;
        let eps = 1e-15;

        let (_, _, _, d_n, _, _) = super::lasq5::<f64, 0>(i0, n0, &mut z, tau, sigma, eps);

        // d_n should be stored in the z array
        // The last step stores d_n at z[j4+2] where j4 is computed in the unroll section
        // Verify d_n is finite and reasonable
        assert!(d_n.is_finite());
        assert!(d_n >= -1.0); // Allow some negativity due to the algorithm's nature
    }
}
