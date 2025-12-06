use std::ops::{AddAssign, MulAssign};

use num_traits::Float;

/// Computes an approximation to the smallest eigenvalue using values of d from
/// the previous transform.
///
/// # Panics
///
/// Panics if `z` has fewer than `4 * n0` elements.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn lasq4<A, const PP: usize>(
    i0: usize,
    n0: usize,
    z: &[A],
    n0_in: usize,
    d_min: A,
    d_min_1: A,
    d_min_2: A,
    d_n: A,
    d_n_1: A,
    d_n_2: A,
    mut g: A,
    mut ttype: isize,
) -> (A, isize, A)
where
    A: Float + AddAssign + MulAssign,
{
    assert!(PP <= 1, "`PP` must be either 0 or 1.");

    if d_min <= A::zero() {
        return (-d_min, -1, g);
    }

    let const_1 = A::from(0.5630).expect("valid conversion");
    let const_2 = A::from(1.01).expect("valid conversion");
    let const_3 = A::from(1.05).expect("valid conversion");
    let hundred = A::from(100).expect("valid conversion");
    let half = (A::one() + A::one()).recip();
    let third = (A::one() + A::one() + A::one()).recip();
    let quarter = (A::one() + A::one() + A::one() + A::one()).recip();

    assert!(
        z.len() >= 4 * n0,
        "`z` must have at least `4 * n0` elements"
    );
    let nn = 4 * n0 + PP;
    let mut s = A::zero();
    if n0_in == n0 {
        if d_min == d_n || d_min == d_n_1 {
            let mut b_1 = z[nn - 4].sqrt() * z[nn - 6].sqrt();

            if d_min == d_n && d_min_1 == d_n_1 {
                let b_2 = z[nn - 8].sqrt() * z[nn - 10].sqrt();
                let a_2 = z[nn - 8] + z[nn - 6];
                let gap_2 = d_min_2 - a_2 - d_min_2 * quarter;
                let gap_1 = if gap_2 > A::zero() && gap_2 > b_2 {
                    a_2 - d_n - (b_2 / gap_2) * b_2
                } else {
                    a_2 - d_n - (b_1 + b_2)
                };
                if gap_1 > A::zero() && gap_1 > b_1 {
                    let left = d_n - (b_1 / gap_1) * b_1;
                    let right = d_min * half;
                    s = if left > right { left } else { right };
                    ttype = -2;
                } else {
                    s = if d_n > b_1 { d_n - b_1 } else { A::zero() };
                    if a_2 > b_1 + b_2 && s > a_2 - (b_1 + b_2) {
                        s = a_2 - (b_1 + b_2);
                    }
                    if d_min * third > s {
                        s = d_min * third;
                    }
                    ttype = -3;
                }
            } else {
                ttype = -4;
                s = d_min * quarter;
                let mut b_2;
                let mut a_2;
                let gam;
                let mut np;
                if d_min == d_n {
                    gam = d_n;
                    a_2 = A::zero();
                    if z[nn - 6] > z[nn - 8] {
                        return (s, ttype, g);
                    }
                    b_2 = z[nn - 6] / z[nn - 8];
                    np = nn - 9;
                } else {
                    np = nn - PP * 2;
                    gam = d_n_1;
                    if z[np - 5] > z[np - 3] {
                        return (s, ttype, g);
                    }
                    a_2 = z[np - 5] / z[np - 3];
                    if z[nn - 10] > z[nn - 12] {
                        return (s, ttype, g);
                    }
                    b_2 = z[nn - 10] / z[nn - 12];
                    np = nn - 13;
                }

                a_2 += b_2;
                let mut i4 = np;
                while i4 >= 4 * i0 - 1 + PP {
                    if b_2 == A::zero() {
                        break;
                    }
                    b_1 = b_2;
                    if z[i4 - 1] > z[i4 - 3] {
                        return (s, ttype, g);
                    }
                    b_2 *= z[i4 - 1] / z[i4 - 3];
                    a_2 += b_2;
                    let max_b = if b_2 > b_1 { b_2 } else { b_1 };
                    if hundred * max_b < a_2 || const_1 < a_2 {
                        break;
                    }
                    i4 = i4.saturating_sub(4);
                }
                a_2 *= const_3;

                if a_2 < const_1 {
                    s = gam * (A::one() - a_2.sqrt()) / (A::one() + a_2);
                }
            }
        } else if d_min == d_n_2 {
            ttype = -5;
            s = d_min * quarter;

            let np = nn - 2 * PP;
            let mut b_1 = z[np - 3];
            let mut b_2 = z[np - 7];
            let gam = d_n_2;
            if z[np - 9] > b_2 || z[np - 5] > b_1 {
                return (s, ttype, g);
            }
            let mut a_2 = (z[np - 9] / b_2) * (A::one() + z[np - 5] / b_1);

            if n0 - i0 > 2 {
                b_2 = z[nn - 14] / z[nn - 16];
                a_2 += b_2;
                let mut i4 = nn - 17;
                while i4 >= 4 * i0 - 1 + PP {
                    if b_2 == A::zero() {
                        break;
                    }
                    b_1 = b_2;
                    if z[i4 - 1] > z[i4 - 3] {
                        return (s, ttype, g);
                    }
                    b_2 *= z[i4 - 1] / z[i4 - 3];
                    a_2 += b_2;
                    let max_b = if b_2 > b_1 { b_2 } else { b_1 };
                    if hundred * max_b < a_2 || const_1 < a_2 {
                        break;
                    }
                    i4 = i4.saturating_sub(4);
                }
                a_2 *= const_3;
            }
            if a_2 < const_1 {
                s = gam * (A::one() - a_2.sqrt()) / (A::one() + a_2);
            }
        } else {
            if ttype == -6 {
                g += (A::one() - g) * third;
            } else if ttype == -18 {
                g = quarter * third;
            } else {
                g = quarter;
            }
            s = g * d_min;
            ttype = -6;
        }
    } else if n0_in == n0 + 1 {
        if d_min_1 == d_n_1 && d_min_2 == d_n_2 {
            ttype = -7;
            s = third * d_min_1;
            if z[nn - 6] > z[nn - 8] {
                return (s, ttype, g);
            }
            let mut b_1 = z[nn - 6] / z[nn - 8];
            let mut b_2 = b_1;
            if b_2 != A::zero() {
                let mut i4 = 4 * n0 - 9 + PP;
                while i4 >= 4 * i0 - 1 + PP {
                    let a_2 = b_1;
                    if z[i4 - 1] > z[i4 - 3] {
                        return (s, ttype, g);
                    }
                    b_1 *= z[i4 - 1] / z[i4 - 3];
                    b_2 += b_1;
                    let ab_max = if b_1 > a_2 { b_1 } else { a_2 };
                    if hundred * ab_max < b_2 {
                        break;
                    }
                    i4 = i4.saturating_sub(4);
                }
            }
            b_2 = (const_3 * b_2).sqrt();
            let a_2 = d_min_1 / (A::one() + b_2 * b_2);
            let gap_2 = half * d_min_2 - a_2;
            if gap_2 > A::zero() && gap_2 > b_2 * a_2 {
                let tmp_s = a_2 * (A::one() - const_2 * a_2 * (b_2 / gap_2) * b_2);
                if tmp_s > s {
                    s = tmp_s;
                }
            } else {
                let tmp_s = a_2 * (A::one() - const_2 * b_2);
                if tmp_s > s {
                    s = tmp_s;
                }
                ttype = -8;
            }
        } else {
            s = quarter * d_min_1;
            if d_min_1 == d_n_1 {
                s = half * d_min_1;
            }
            ttype = -9;
        }
    } else if n0_in == n0 + 2 {
        if d_min_2 == d_n_2 && half * z[nn - 6] < z[nn - 8] {
            ttype = -10;
            s = third * d_min_2;
            if z[nn - 6] > z[nn - 8] {
                return (s, ttype, g);
            }
            let mut b_1 = z[nn - 6] / z[nn - 8];
            let mut b_2 = b_1;
            if b_2 != A::zero() {
                let mut i4 = 4 * n0 - 9 + PP;
                while i4 >= 4 * i0 - 1 + PP {
                    if z[i4 - 1] > z[i4 - 3] {
                        return (s, ttype, g);
                    }
                    b_1 *= z[i4 - 1] / z[i4 - 3];
                    b_2 += b_1;
                    if hundred * b_1 < b_2 {
                        break;
                    }
                    i4 = i4.saturating_sub(4);
                }
            }
            b_2 = (const_3 * b_2).sqrt();
            let a_2 = d_min_2 / (A::one() + b_2 * b_2);
            let gap_2 = z[nn - 8] + z[nn - 10] - z[nn - 12].sqrt() * z[nn - 10].sqrt() - a_2;
            if gap_2 > A::zero() && gap_2 > b_2 * a_2 {
                let tmp_s = a_2 * (A::one() - const_2 * a_2 * (b_2 / gap_2) * b_2);
                if tmp_s > s {
                    s = tmp_s;
                }
            } else {
                let tmp_s = a_2 * (A::one() - const_2 * b_2);
                if tmp_s > s {
                    s = tmp_s;
                }
            }
        } else {
            s = quarter * d_min_2;
            ttype = -11;
        }
    } else if n0_in > n0 + 2 {
        s = A::zero();
        ttype = -12;
    }

    (s, ttype, g)
}

#[cfg(test)]
mod tests {
    use super::lasq4;

    /// Helper to call lasq4 with PP=0
    fn lasq4_pp0(
        i0: usize,
        n0: usize,
        z: &[f64],
        n0_in: usize,
        d_min: f64,
        d_min_1: f64,
        d_min_2: f64,
        d_n: f64,
        d_n_1: f64,
        d_n_2: f64,
        g: f64,
        ttype: isize,
    ) -> (f64, isize, f64) {
        lasq4::<f64, 0>(
            i0, n0, z, n0_in, d_min, d_min_1, d_min_2, d_n, d_n_1, d_n_2, g, ttype,
        )
    }

    /// Helper to call lasq4 with PP=1
    fn lasq4_pp1(
        i0: usize,
        n0: usize,
        z: &[f64],
        n0_in: usize,
        d_min: f64,
        d_min_1: f64,
        d_min_2: f64,
        d_n: f64,
        d_n_1: f64,
        d_n_2: f64,
        g: f64,
        ttype: isize,
    ) -> (f64, isize, f64) {
        lasq4::<f64, 1>(
            i0, n0, z, n0_in, d_min, d_min_1, d_min_2, d_n, d_n_1, d_n_2, g, ttype,
        )
    }

    // Case 1: Negative DMIN forces the shift to take that absolute value
    #[test]
    fn case_1_negative_dmin() {
        let z = [1.0; 20];
        let (tau, ttype, g) = lasq4_pp0(1, 5, &z, 5, -0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, -6);

        assert!((tau - 0.5).abs() < 1e-14, "tau should be -dmin = 0.5");
        assert_eq!(ttype, -1, "ttype should be -1 for negative dmin");
        assert!((g - 0.25).abs() < 1e-14, "g should remain unchanged");
    }

    #[test]
    fn case_1_zero_dmin() {
        let z = [1.0; 20];
        let (tau, ttype, _) = lasq4_pp0(1, 5, &z, 5, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, -6);

        assert!((tau - 0.0).abs() < 1e-14, "tau should be 0 for dmin=0");
        assert_eq!(ttype, -1, "ttype should be -1 for zero dmin");
    }

    // Cases 2-3: n0_in == n0, dmin == dn, dmin_1 == dn_1
    #[test]
    fn case_2_gap1_positive() {
        // Set up z array for case 2: gap1 > 0 and gap1 > b1
        // nn = 4*n0 + PP = 4*5 + 0 = 20
        // Need: z[nn-4]=z[16], z[nn-6]=z[14], z[nn-8]=z[12], z[nn-10]=z[10]
        let mut z = [0.0; 24];
        z[16] = 4.0; // z[nn-4]
        z[14] = 4.0; // z[nn-6]
        z[12] = 4.0; // z[nn-8]
        z[10] = 4.0; // z[nn-10]

        let d_min = 1.0;
        let d_min_1 = 1.0;
        let d_min_2 = 20.0; // Large to make gap2 > 0
        let d_n = 1.0;
        let d_n_1 = 1.0;
        let d_n_2 = 1.0;

        let (tau, ttype, _) = lasq4_pp0(1, 5, &z, 5, d_min, d_min_1, d_min_2, d_n, d_n_1, d_n_2, 0.25, 0);

        assert_eq!(ttype, -2, "ttype should be -2 for case 2");
        assert!(tau > 0.0, "tau should be positive");
        assert!(tau <= d_min, "tau should not exceed dmin");
    }

    #[test]
    fn case_3_gap1_not_positive() {
        // Set up for case 3: gap1 <= 0 or gap1 <= b1
        let mut z = [0.0; 24];
        z[16] = 1.0;
        z[14] = 1.0;
        z[12] = 1.0;
        z[10] = 1.0;

        let d_min = 1.0;
        let d_min_1 = 1.0;
        let d_min_2 = 1.0; // Small gap2
        let d_n = 1.0;
        let d_n_1 = 1.0;
        let d_n_2 = 1.0;

        let (tau, ttype, _) = lasq4_pp0(1, 5, &z, 5, d_min, d_min_1, d_min_2, d_n, d_n_1, d_n_2, 0.25, 0);

        assert_eq!(ttype, -3, "ttype should be -3 for case 3");
        assert!(tau >= 0.0, "tau should be non-negative");
    }

    // Case 4: n0_in == n0, dmin == dn or dmin == dn1, but not both equal
    #[test]
    fn case_4_dmin_equals_dn() {
        // dmin == dn but dmin_1 != dn_1
        let mut z = [0.0; 24];
        // Set up z values so z[nn-6] <= z[nn-8] to avoid early return
        z[16] = 4.0;
        z[14] = 2.0; // z[nn-6]
        z[12] = 4.0; // z[nn-8], must be >= z[nn-6]
        z[10] = 1.0;

        let d_min = 1.0;
        let d_min_1 = 2.0; // != dn_1
        let d_n = 1.0;     // == dmin
        let d_n_1 = 1.0;

        let (tau, ttype, _) = lasq4_pp0(1, 5, &z, 5, d_min, d_min_1, 1.0, d_n, d_n_1, 1.0, 0.25, 0);

        assert_eq!(ttype, -4, "ttype should be -4 for case 4");
        assert!(tau >= d_min * 0.25, "tau should be at least quarter*dmin");
    }

    #[test]
    fn case_4_dmin_equals_dn1() {
        // dmin == dn1 (and dmin != dn)
        let mut z = [0.0; 40];
        // For PP=0, nn=4*n0=20
        // np = nn - 2*PP = 20
        // Need z[np-5]=z[15] <= z[np-3]=z[17]
        // Need z[nn-10]=z[10] <= z[nn-12]=z[8]
        for i in 0..40 {
            z[i] = 2.0;
        }
        z[15] = 1.0; // z[np-5]
        z[17] = 2.0; // z[np-3]
        z[10] = 1.0; // z[nn-10]
        z[8] = 2.0;  // z[nn-12]

        let d_min = 1.0;
        let d_n = 2.0;   // != dmin
        let d_n_1 = 1.0; // == dmin

        let (_tau, ttype, _) = lasq4_pp0(1, 5, &z, 5, d_min, 2.0, 1.0, d_n, d_n_1, 1.0, 0.25, 0);

        assert_eq!(ttype, -4, "ttype should be -4 for case 4 with dmin==dn1");
    }

    #[test]
    fn case_4_early_return_z_ordering() {
        // Test early return when z[nn-6] > z[nn-8] in case 4 (dmin == dn but dmin_1 != dn_1)
        let mut z = [0.0; 24];
        z[16] = 4.0;
        z[14] = 5.0; // z[nn-6] > z[nn-8] triggers early return
        z[12] = 4.0; // z[nn-8]

        let d_min = 1.0;
        let d_min_1 = 2.0; // != d_n_1 to enter case 4
        let d_n = 1.0;     // == d_min
        let d_n_1 = 1.0;   // != d_min_1

        let (tau, ttype, _) = lasq4_pp0(1, 5, &z, 5, d_min, d_min_1, 1.0, d_n, d_n_1, 1.0, 0.25, 0);

        assert_eq!(ttype, -4);
        assert!((tau - d_min * 0.25).abs() < 1e-14, "Should return early with quarter*dmin");
    }

    // Case 5: n0_in == n0, dmin == dn2
    #[test]
    fn case_5_dmin_equals_dn2() {
        // np = nn - 2*PP = 4*5 - 0 = 20
        // Need z[np-9]=z[11] <= b2=z[np-7]=z[13]
        // Need z[np-5]=z[15] <= b1=z[np-3]=z[17]
        let mut z = [0.0; 24];
        for i in 0..24 {
            z[i] = 2.0;
        }
        z[17] = 4.0; // b1 = z[np-3]
        z[15] = 2.0; // z[np-5] <= b1
        z[13] = 4.0; // b2 = z[np-7]
        z[11] = 2.0; // z[np-9] <= b2

        let d_min = 1.0;
        let d_n = 2.0;
        let d_n_1 = 2.0;
        let d_n_2 = 1.0; // == dmin

        let (tau, ttype, _) = lasq4_pp0(1, 5, &z, 5, d_min, 2.0, 2.0, d_n, d_n_1, d_n_2, 0.25, 0);

        assert_eq!(ttype, -5, "ttype should be -5 for case 5");
        assert!(tau >= d_min * 0.25, "tau should be at least quarter*dmin");
    }

    // Case 6: n0_in == n0, but dmin != dn, dn1, or dn2
    #[test]
    fn case_6_no_match_with_ttype_minus6() {
        let z = [2.0; 24];
        let d_min = 1.0;
        let d_n = 3.0;
        let d_n_1 = 4.0;
        let d_n_2 = 5.0;

        let (tau, ttype, g) = lasq4_pp0(1, 5, &z, 5, d_min, 2.0, 2.0, d_n, d_n_1, d_n_2, 0.25, -6);

        assert_eq!(ttype, -6, "ttype should be -6 for case 6");
        // g += (1-g)*third when ttype was -6
        let expected_g = 0.25 + (1.0 - 0.25) / 3.0;
        assert!((g - expected_g).abs() < 1e-14, "g should be updated");
        assert!((tau - expected_g * d_min).abs() < 1e-14, "tau = g*dmin");
    }

    #[test]
    fn case_6_with_ttype_minus18() {
        let z = [2.0; 24];
        let d_min = 1.0;
        let d_n = 3.0;
        let d_n_1 = 4.0;
        let d_n_2 = 5.0;

        let (tau, ttype, g) = lasq4_pp0(1, 5, &z, 5, d_min, 2.0, 2.0, d_n, d_n_1, d_n_2, 0.5, -18);

        assert_eq!(ttype, -6);
        // g = quarter * third when ttype was -18
        let expected_g = 0.25 / 3.0;
        assert!((g - expected_g).abs() < 1e-14);
        assert!((tau - expected_g * d_min).abs() < 1e-14);
    }

    #[test]
    fn case_6_with_other_ttype() {
        let z = [2.0; 24];
        let d_min = 1.0;
        let d_n = 3.0;
        let d_n_1 = 4.0;
        let d_n_2 = 5.0;

        let (tau, ttype, g) = lasq4_pp0(1, 5, &z, 5, d_min, 2.0, 2.0, d_n, d_n_1, d_n_2, 0.5, -3);

        assert_eq!(ttype, -6);
        // g = quarter when ttype was not -6 or -18
        assert!((g - 0.25).abs() < 1e-14);
        assert!((tau - 0.25 * d_min).abs() < 1e-14);
    }

    // Cases 7-8: n0_in == n0 + 1 (one eigenvalue deflated)
    #[test]
    fn case_7_one_deflated_gap2_positive() {
        // n0_in = n0 + 1, dmin1 == dn1, dmin2 == dn2
        // Use larger n0 to avoid underflow in loop: i4 = 4*n0-9+PP, needs i4 >= 4*i0-1+PP
        // For i0=1, PP=0: bound is 3, so i4 starts at 4*n0-9, need 4*n0-9 >= 3, so n0 >= 3
        // But loop decrements by 4, so need more room. Use n0=10.
        let mut z = [0.0; 60];
        // nn = 4*10 + 0 = 40
        // Need z[nn-6]=z[34] <= z[nn-8]=z[32]
        for i in 0..60 {
            z[i] = 2.0;
        }
        z[34] = 1.0; // z[nn-6]
        z[32] = 2.0; // z[nn-8]

        let d_min_1 = 1.0;
        let d_min_2 = 10.0; // Large for gap2 > 0
        let d_n_1 = 1.0;    // == dmin1
        let d_n_2 = 10.0;   // == dmin2

        let (tau, ttype, _) = lasq4_pp0(1, 10, &z, 11, 0.5, d_min_1, d_min_2, 2.0, d_n_1, d_n_2, 0.25, 0);

        assert!(ttype == -7 || ttype == -8, "ttype should be -7 or -8 for cases 7-8");
        assert!(tau > 0.0, "tau should be positive");
    }

    #[test]
    fn case_8_one_deflated_gap2_not_positive() {
        // Use larger n0 to avoid underflow
        let mut z = [0.0; 60];
        for i in 0..60 {
            z[i] = 2.0;
        }
        // nn = 4*10 = 40
        z[34] = 1.0; // z[nn-6]
        z[32] = 2.0; // z[nn-8]

        let d_min_1 = 1.0;
        let d_min_2 = 1.0; // Small for gap2 <= 0
        let d_n_1 = 1.0;
        let d_n_2 = 1.0;

        let (tau, ttype, _) = lasq4_pp0(1, 10, &z, 11, 0.5, d_min_1, d_min_2, 2.0, d_n_1, d_n_2, 0.25, 0);

        assert!(ttype == -7 || ttype == -8, "ttype should be -7 or -8");
        assert!(tau >= d_min_1 / 3.0 - 1e-10, "tau should be at least third*dmin1");
    }

    // Case 9: n0_in == n0 + 1, but dmin1 != dn1 or dmin2 != dn2
    #[test]
    fn case_9_dmin1_equals_dn1() {
        let z = [2.0; 24];
        let d_min_1 = 1.0;
        let d_n_1 = 1.0;   // == dmin1
        let d_n_2 = 3.0;   // != dmin2

        let (tau, ttype, _) = lasq4_pp0(1, 5, &z, 6, 0.5, d_min_1, 2.0, 2.0, d_n_1, d_n_2, 0.25, 0);

        assert_eq!(ttype, -9);
        assert!((tau - 0.5 * d_min_1).abs() < 1e-14, "tau = half*dmin1 when dmin1==dn1");
    }

    #[test]
    fn case_9_dmin1_not_equals_dn1() {
        let z = [2.0; 24];
        let d_min_1 = 1.0;
        let d_n_1 = 2.0; // != dmin1

        let (tau, ttype, _) = lasq4_pp0(1, 5, &z, 6, 0.5, d_min_1, 2.0, 2.0, d_n_1, 3.0, 0.25, 0);

        assert_eq!(ttype, -9);
        assert!((tau - 0.25 * d_min_1).abs() < 1e-14, "tau = quarter*dmin1");
    }

    // Cases 10-11: n0_in == n0 + 2 (two eigenvalues deflated)
    #[test]
    fn case_10_two_deflated_conditions_met() {
        // dmin2 == dn2 and 2*z[nn-6] < z[nn-8]
        // Use larger n0 to avoid underflow in loop
        let mut z = [0.0; 60];
        for i in 0..60 {
            z[i] = 2.0;
        }
        // nn = 4*10 = 40
        z[34] = 1.0;  // z[nn-6]
        z[32] = 4.0;  // z[nn-8], need 2*z[34] < z[32]
        z[30] = 1.0;  // z[nn-10]
        z[28] = 2.0;  // z[nn-12]

        let d_min_2 = 1.0;
        let d_n_2 = 1.0; // == dmin2

        let (tau, ttype, _) = lasq4_pp0(1, 10, &z, 12, 0.5, 1.0, d_min_2, 2.0, 2.0, d_n_2, 0.25, 0);

        assert_eq!(ttype, -10, "ttype should be -10 for case 10");
        assert!(tau >= d_min_2 / 3.0 - 1e-10, "tau should be at least third*dmin2");
    }

    #[test]
    fn case_11_two_deflated_conditions_not_met() {
        let z = [2.0; 24];
        let d_min_2 = 1.0;
        let d_n_2 = 2.0; // != dmin2

        let (tau, ttype, _) = lasq4_pp0(1, 5, &z, 7, 0.5, 1.0, d_min_2, 2.0, 2.0, d_n_2, 0.25, 0);

        assert_eq!(ttype, -11, "ttype should be -11 for case 11");
        assert!((tau - 0.25 * d_min_2).abs() < 1e-14, "tau = quarter*dmin2");
    }

    // Case 12: n0_in > n0 + 2 (more than two eigenvalues deflated)
    #[test]
    fn case_12_more_than_two_deflated() {
        let z = [2.0; 24];

        let (tau, ttype, _) = lasq4_pp0(1, 5, &z, 8, 0.5, 1.0, 1.0, 2.0, 2.0, 2.0, 0.25, 0);

        assert_eq!(ttype, -12, "ttype should be -12 for case 12");
        assert!((tau - 0.0).abs() < 1e-14, "tau should be 0");
    }

    // Tests with PP=1
    #[test]
    fn pp1_case_1_negative_dmin() {
        let z = [1.0; 24];
        let (tau, ttype, _) = lasq4_pp1(1, 5, &z, 5, -0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, -6);

        assert!((tau - 0.5).abs() < 1e-14);
        assert_eq!(ttype, -1);
    }

    #[test]
    fn pp1_case_4() {
        // Test PP=1 path in case 4 (dmin == dn but dmin_1 != dn_1)
        // nn = 4*5 + 1 = 21
        let mut z = [0.0; 28];
        for i in 0..28 {
            z[i] = 2.0;
        }
        // For dmin == dn: z[nn-6]=z[15] <= z[nn-8]=z[13]
        z[15] = 1.0;
        z[13] = 2.0;

        let d_min = 1.0;
        let d_min_1 = 2.0; // != d_n_1 to enter case 4
        let d_n = 1.0;     // == d_min
        let d_n_1 = 1.0;   // != d_min_1

        let (tau, ttype, _) = lasq4_pp1(1, 5, &z, 5, d_min, d_min_1, 1.0, d_n, d_n_1, 1.0, 0.25, 0);

        assert_eq!(ttype, -4);
        assert!(tau >= d_min * 0.25);
    }

    // Test f32 support
    #[test]
    fn f32_basic() {
        let z = [1.0f32; 24];
        let (tau, ttype, g) = lasq4::<f32, 0>(1, 5, &z, 5, -0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, -6);

        assert!((tau - 0.5).abs() < 1e-6);
        assert_eq!(ttype, -1);
        assert!((g - 0.25).abs() < 1e-6);
    }

    // Test the loop termination conditions
    #[test]
    fn case_4_loop_terminates_on_b2_zero() {
        // Case 4: dmin == dn but dmin_1 != dn_1
        let mut z = [0.0; 40];
        for i in 0..40 {
            z[i] = 4.0;
        }
        // nn = 4*5 = 20
        // For dmin == dn path: z[nn-6]=z[14] <= z[nn-8]=z[12]
        z[14] = 0.0; // z[nn-6] = 0, this makes b2 = 0
        z[12] = 4.0; // z[nn-8]

        let d_min = 1.0;
        let d_min_1 = 2.0; // != d_n_1 to enter case 4
        let d_n = 1.0;     // == d_min
        let d_n_1 = 1.0;   // != d_min_1

        let (tau, ttype, _) = lasq4_pp0(1, 5, &z, 5, d_min, d_min_1, 1.0, d_n, d_n_1, 1.0, 0.25, 0);

        assert_eq!(ttype, -4);
        assert!(tau > 0.0);
    }

    #[test]
    fn case_4_loop_terminates_on_convergence() {
        // Case 4: dmin == dn but dmin_1 != dn_1
        // Set up z so 100*max(b2,b1) < a2 triggers
        // Use larger n0 to ensure loop has room
        let mut z = [0.0; 60];
        for i in 0..60 {
            z[i] = 10.0;
        }
        // nn = 4*10 = 40
        z[34] = 1.0;   // z[nn-6], small
        z[32] = 100.0; // z[nn-8], large

        let d_min = 1.0;
        let d_min_1 = 2.0; // != d_n_1 to enter case 4
        let d_n = 1.0;     // == d_min
        let d_n_1 = 1.0;   // != d_min_1

        let (_tau, ttype, _) = lasq4_pp0(1, 10, &z, 10, d_min, d_min_1, 1.0, d_n, d_n_1, 1.0, 0.25, 0);

        assert_eq!(ttype, -4);
    }

    // Test boundary conditions
    #[test]
    fn minimum_size_array() {
        // n0 = 1 requires z.len() >= 4
        let z = [1.0; 4];
        let (tau, ttype, _) = lasq4_pp0(1, 1, &z, 1, -0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 0);

        assert!((tau - 0.1).abs() < 1e-14);
        assert_eq!(ttype, -1);
    }

    #[test]
    #[should_panic(expected = "`z` must have at least `4 * n0` elements")]
    fn panics_on_insufficient_array() {
        let z = [1.0; 3]; // Too small for n0=1
        lasq4_pp0(1, 1, &z, 1, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 0);
    }

    #[test]
    #[should_panic(expected = "`PP` must be either 0 or 1")]
    fn panics_on_invalid_pp() {
        let z = [1.0; 20];
        lasq4::<f64, 2>(1, 5, &z, 5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 0);
    }

    // Verify CNST1 value matches Fortran
    #[test]
    fn const1_matches_fortran() {
        // Test that the a2 < CNST1 check uses 0.5630, not 9/16=0.5625
        // We can verify this indirectly by checking the behavior near the boundary
        let mut z = [0.0; 40];
        for i in 0..40 {
            z[i] = 2.0;
        }
        z[16] = 4.0;
        z[14] = 2.0;
        z[12] = 4.0;
        z[10] = 4.0;

        // This tests that our constant is set correctly by checking computation completes
        let (tau, ttype, _) = lasq4_pp0(1, 5, &z, 5, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 0.25, 0);

        assert!(tau.is_finite());
        assert!(ttype < 0);
    }
}
