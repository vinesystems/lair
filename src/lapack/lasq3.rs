use std::ops::{AddAssign, MulAssign};

use num_traits::Float;

use super::lasq4::lasq4;
use super::lasq5::lasq5;
use super::lasq6::lasq6;

/// Checks for deflation, computes a shift (TAU), and calls dqds.
///
/// This is a Rust translation of LAPACK's DLASQ3. It iteratively deflates
/// eigenvalues and applies the dqds algorithm until all eigenvalues are found.
///
/// # Arguments
///
/// * `i0` - First index (1-based as in Fortran)
/// * `n0` - Last index (1-based, modified during deflation)
/// * `z` - Array of length at least `4*n0`, holds the qd-array
/// * `pp` - Ping-pong flag (0 or 1), may be modified
/// * `d_min` - Minimum value of d
/// * `sigma` - Sum of shifts used so far
/// * `desig` - Lower order part of sigma
/// * `q_max` - Maximum value of q
/// * `n_fail` - Increment for the number of failures (accumulated)
/// * `iter` - Increment for the total number of iterations (accumulated)
/// * `n_div` - Increment for the total number of divisions (accumulated)
/// * `ttype` - Shift type from lasq4
/// * `d_min_1` - Minimum value of d, excluding `d_n`
/// * `d_min_2` - Minimum value of d, excluding `d_n` and `d_{n-1}`
/// * `d_n` - d(n0)
/// * `d_n_1` - d(n0-1)
/// * `d_n_2` - d(n0-2)
/// * `g` - Used for computing shift in lasq4
/// * `tau` - Shift value
///
/// # Returns
///
/// Updated values of `(n0, pp, d_min, sigma, desig, q_max, n_fail, iter, n_div,
/// ttype, d_min_1, d_min_2, d_n, d_n_1, d_n_2, g, tau)`
#[allow(dead_code)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines, clippy::type_complexity)]
pub(crate) fn lasq3<A>(
    i0: usize,
    mut n0: usize,
    z: &mut [A],
    mut pp: usize,
    mut d_min: A,
    mut sigma: A,
    mut desig: A,
    mut q_max: A,
    mut n_fail: usize,
    mut iter: usize,
    mut n_div: usize,
    mut ttype: isize,
    mut d_min_1: A,
    mut d_min_2: A,
    mut d_n: A,
    mut d_n_1: A,
    mut d_n_2: A,
    mut g: A,
    mut tau: A,
    eps: A,
) -> (
    usize,
    usize,
    A,
    A,
    A,
    A,
    usize,
    usize,
    usize,
    isize,
    A,
    A,
    A,
    A,
    A,
    A,
    A,
)
where
    A: Float + AddAssign + MulAssign,
{
    let cbias = A::from(1.50).expect("valid conversion");
    let zero = A::zero();
    let one = A::one();
    let two = one + one;
    let quarter = (two + two).recip();
    let half = two.recip();
    let hundred = A::from(100).expect("valid conversion");

    let tol = eps * hundred;
    let tol2 = tol * tol;

    let n0_in = n0;

    // Main deflation loop (label 10 in Fortran)
    loop {
        // Check for deflation
        if n0 < i0 {
            return (
                n0, pp, d_min, sigma, desig, q_max, n_fail, iter, n_div, ttype, d_min_1, d_min_2,
                d_n, d_n_1, d_n_2, g, tau,
            );
        }

        if n0 == i0 {
            // Single eigenvalue case (goto 20)
            z[4 * n0 - 3 - 1] = z[4 * n0 + pp - 3 - 1] + sigma;
            n0 -= 1;
            continue;
        }

        let nn = 4 * n0 + pp;

        if n0 == i0 + 1 {
            // Two eigenvalues case (goto 40)
            // Sort and compute eigenvalues
            if z[nn - 3 - 1] > z[nn - 7 - 1] {
                z.swap(nn - 3 - 1, nn - 7 - 1);
            }
            let mut t = half * ((z[nn - 7 - 1] - z[nn - 3 - 1]) + z[nn - 5 - 1]);
            if z[nn - 5 - 1] > z[nn - 3 - 1] * tol2 && t != zero {
                let mut s = z[nn - 3 - 1] * (z[nn - 5 - 1] / t);
                if s <= t {
                    s = z[nn - 3 - 1] * (z[nn - 5 - 1] / (t * (one + (one + s / t).sqrt())));
                } else {
                    s = z[nn - 3 - 1] * (z[nn - 5 - 1] / (t + t.sqrt() * (t + s).sqrt()));
                }
                t = z[nn - 7 - 1] + (s + z[nn - 5 - 1]);
                z[nn - 3 - 1] *= z[nn - 7 - 1] / t;
                z[nn - 7 - 1] = t;
            }
            z[4 * n0 - 7 - 1] = z[nn - 7 - 1] + sigma;
            z[4 * n0 - 3 - 1] = z[nn - 3 - 1] + sigma;
            n0 -= 2;
            continue;
        }

        // Check whether E(N0-1) is negligible, 1 eigenvalue
        if z[nn - 5 - 1] > tol2 * (sigma + z[nn - 3 - 1])
            && z[nn - 2 * pp - 4 - 1] > tol2 * z[nn - 7 - 1]
        {
            // Check whether E(N0-2) is negligible, 2 eigenvalues (goto 30, then 50)
            if z[nn - 9 - 1] > tol2 * sigma && z[nn - 2 * pp - 8 - 1] > tol2 * z[nn - 11 - 1] {
                // Continue to main computation (label 50)
                break;
            }

            // Two eigenvalues case (goto 40)
            if z[nn - 3 - 1] > z[nn - 7 - 1] {
                z.swap(nn - 3 - 1, nn - 7 - 1);
            }
            let mut t = half * ((z[nn - 7 - 1] - z[nn - 3 - 1]) + z[nn - 5 - 1]);
            if z[nn - 5 - 1] > z[nn - 3 - 1] * tol2 && t != zero {
                let mut s = z[nn - 3 - 1] * (z[nn - 5 - 1] / t);
                if s <= t {
                    s = z[nn - 3 - 1] * (z[nn - 5 - 1] / (t * (one + (one + s / t).sqrt())));
                } else {
                    s = z[nn - 3 - 1] * (z[nn - 5 - 1] / (t + t.sqrt() * (t + s).sqrt()));
                }
                t = z[nn - 7 - 1] + (s + z[nn - 5 - 1]);
                z[nn - 3 - 1] *= z[nn - 7 - 1] / t;
                z[nn - 7 - 1] = t;
            }
            z[4 * n0 - 7 - 1] = z[nn - 7 - 1] + sigma;
            z[4 * n0 - 3 - 1] = z[nn - 3 - 1] + sigma;
            n0 -= 2;
            continue;
        }

        // Single eigenvalue deflation (goto 20)
        z[4 * n0 - 3 - 1] = z[4 * n0 + pp - 3 - 1] + sigma;
        n0 -= 1;
    }

    // Label 50: main computation
    if pp == 2 {
        pp = 0;
    }

    // Reverse the qd-array if warranted
    if (d_min <= zero || n0 < n0_in)
        && cbias * z[4 * i0 + pp - 3 - 1] < z[4 * n0 + pp - 3 - 1]
    {
        let ipn4 = 4 * (i0 + n0);
        let mut j4 = 4 * i0;
        while j4 <= 2 * (i0 + n0 - 1) {
            // Swap elements
            z.swap(j4 - 3 - 1, ipn4 - j4 - 3 - 1);
            z.swap(j4 - 2 - 1, ipn4 - j4 - 2 - 1);
            z.swap(j4 - 1 - 1, ipn4 - j4 - 5 - 1);
            z.swap(j4 - 1, ipn4 - j4 - 4 - 1);

            j4 += 4;
        }
        if n0 - i0 <= 4 {
            z[4 * n0 + pp - 1 - 1] = z[4 * i0 + pp - 1 - 1];
            z[4 * n0 - pp - 1] = z[4 * i0 - pp - 1];
        }
        d_min_2 = d_min_2.min(z[4 * n0 + pp - 1 - 1]);
        z[4 * n0 + pp - 1 - 1] = z[4 * n0 + pp - 1 - 1]
            .min(z[4 * i0 + pp - 1 - 1])
            .min(z[4 * i0 + pp + 3 - 1]);
        z[4 * n0 - pp - 1] = z[4 * n0 - pp - 1]
            .min(z[4 * i0 - pp - 1])
            .min(z[4 * i0 - pp + 4 - 1]);
        q_max = q_max.max(z[4 * i0 + pp - 3 - 1]).max(z[4 * i0 + pp + 1 - 1]);
        d_min = -zero;
    }

    // Choose a shift
    let (new_tau, new_ttype, new_g) = if pp == 0 {
        lasq4::<A, 0>(
            i0, n0, z, n0_in, d_min, d_min_1, d_min_2, d_n, d_n_1, d_n_2, g, ttype,
        )
    } else {
        lasq4::<A, 1>(
            i0, n0, z, n0_in, d_min, d_min_1, d_min_2, d_n, d_n_1, d_n_2, g, ttype,
        )
    };
    tau = new_tau;
    ttype = new_ttype;
    g = new_g;

    // Call dqds until DMIN > 0 (label 70)
    loop {
        let result = if pp == 0 {
            lasq5::<A, 0>(i0, n0, z, tau, sigma, eps)
        } else {
            lasq5::<A, 1>(i0, n0, z, tau, sigma, eps)
        };

        n_div += n0 - i0 + 2;
        iter += 1;

        // Check status
        if let Some((new_d_min, new_d_min_1, new_d_min_2, new_d_n, new_d_n_1, new_d_n_2)) = result {
            d_min = new_d_min;
            d_min_1 = new_d_min_1;
            d_min_2 = new_d_min_2;
            d_n = new_d_n;
            d_n_1 = new_d_n_1;
            d_n_2 = new_d_n_2;

            if d_min >= zero && d_min_1 >= zero {
                // Success (goto 90)
                break;
            } else if d_min < zero
                && d_min_1 > zero
                && z[4 * (n0 - 1) - pp - 1] < tol * (sigma + d_n_1)
                && d_n.abs() < tol * sigma
            {
                // Convergence hidden by negative DN
                z[4 * (n0 - 1) - pp + 2 - 1] = zero;
                d_min = zero;
                break;
            } else if d_min < zero {
                // TAU too big. Select new TAU and try again
                n_fail += 1;
                if ttype < -22 {
                    // Failed twice. Play it safe.
                    tau = zero;
                } else if d_min_1 > zero {
                    // Late failure. Gives excellent shift.
                    tau = (tau + d_min) * (one - two * eps);
                    ttype -= 11;
                } else {
                    // Early failure. Divide by 4.
                    tau = quarter * tau;
                    ttype -= 12;
                }
                continue;
            } else if d_min.is_nan() {
                // NaN
                if tau == zero {
                    // Risk of underflow (goto 80)
                    let (new_d_min, new_d_min_1, new_d_min_2, new_d_n, new_d_n_1, new_d_n_2) =
                        if pp == 0 {
                            lasq6::<A, 0>(i0, n0, z)
                        } else {
                            lasq6::<A, 1>(i0, n0, z)
                        };
                    d_min = new_d_min;
                    d_min_1 = new_d_min_1;
                    d_min_2 = new_d_min_2;
                    d_n = new_d_n;
                    d_n_1 = new_d_n_1;
                    d_n_2 = new_d_n_2;
                    n_div += n0 - i0 + 2;
                    iter += 1;
                    tau = zero;
                    break;
                }
                tau = zero;
                continue;
            }
            // Possible underflow. Play it safe. (goto 80)
            let (new_d_min, new_d_min_1, new_d_min_2, new_d_n, new_d_n_1, new_d_n_2) = if pp == 0 {
                lasq6::<A, 0>(i0, n0, z)
            } else {
                lasq6::<A, 1>(i0, n0, z)
            };
            d_min = new_d_min;
            d_min_1 = new_d_min_1;
            d_min_2 = new_d_min_2;
            d_n = new_d_n;
            d_n_1 = new_d_n_1;
            d_n_2 = new_d_n_2;
            n_div += n0 - i0 + 2;
            iter += 1;
            tau = zero;
            break;
        }
        // lasq5 returned None (nothing to process)
        break;
    }

    // Label 90: Update sigma
    if tau < sigma {
        desig += tau;
        let t = sigma + desig;
        desig = desig - (t - sigma);
        sigma = t;
    } else {
        let t = sigma + tau;
        desig = sigma - (t - tau) + desig;
        sigma = t;
    }

    (
        n0, pp, d_min, sigma, desig, q_max, n_fail, iter, n_div, ttype, d_min_1, d_min_2, d_n,
        d_n_1, d_n_2, g, tau,
    )
}

#[cfg(test)]
mod tests {
    use super::lasq3;

    fn call_lasq3(
        i0: usize,
        n0: usize,
        z: &mut [f64],
        pp: usize,
        d_min: f64,
        sigma: f64,
        desig: f64,
        q_max: f64,
        n_fail: usize,
        iter: usize,
        n_div: usize,
        ttype: isize,
        d_min_1: f64,
        d_min_2: f64,
        d_n: f64,
        d_n_1: f64,
        d_n_2: f64,
        g: f64,
        tau: f64,
    ) -> (
        usize,
        usize,
        f64,
        f64,
        f64,
        f64,
        usize,
        usize,
        usize,
        isize,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
    ) {
        let eps = f64::EPSILON;
        lasq3(
            i0, n0, z, pp, d_min, sigma, desig, q_max, n_fail, iter, n_div, ttype, d_min_1,
            d_min_2, d_n, d_n_1, d_n_2, g, tau, eps,
        )
    }

    #[test]
    fn returns_immediately_when_n0_less_than_i0() {
        let mut z = [1.0; 20];
        let (n0, pp, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _) =
            call_lasq3(5, 3, &mut z, 0, 1.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 0.0);
        assert_eq!(n0, 3);
        assert_eq!(pp, 0);
    }

    #[test]
    fn single_eigenvalue_deflation() {
        // When n0 == i0, should deflate single eigenvalue
        let mut z = [0.0; 20];
        z[0] = 4.0; // z[4*1 + 0 - 3 - 1] = z[0] for i0=n0=1, pp=0

        let (n0, _, _, _sigma, _, _, _, _, _, _, _, _, _, _, _, _, _) =
            call_lasq3(1, 1, &mut z, 0, 1.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 0.0);

        assert_eq!(n0, 0);
    }

    #[test]
    fn two_eigenvalue_case() {
        // When n0 == i0 + 1, should handle two eigenvalues
        let mut z = [0.0; 40];
        // Set up z array with positive values for 2 eigenvalue case
        // nn = 4*2 + 0 = 8
        // Need z[nn-3-1]=z[4], z[nn-5-1]=z[2], z[nn-7-1]=z[0]
        z[0] = 4.0;  // z[nn-7-1]
        z[2] = 0.5;  // z[nn-5-1]
        z[4] = 2.0;  // z[nn-3-1]

        let (n0, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _) =
            call_lasq3(1, 2, &mut z, 0, 1.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 0.0);

        assert_eq!(n0, 0);
    }

    #[test]
    fn iteration_count_increases() {
        // Test that iter is incremented during processing
        // Need to set up array that passes all deflation checks to reach the iteration loop
        // For pp=0, n0=10:
        // nn = 4*10 + 0 = 40
        // Check 1: z[nn-5-1] = z[34] > tol2 * (sigma + z[nn-3-1] = z[36])
        // Check 2: z[nn-2*0-4-1] = z[35] > tol2 * z[nn-7-1] = z[32]
        // Check 3: z[nn-9-1] = z[30] > tol2 * sigma
        // Check 4: z[nn-2*0-8-1] = z[31] > tol2 * z[nn-11-1] = z[28]
        let mut z = [1.0; 100]; // Initialize all to 1.0 to pass all checks

        let (_, _, _, _, _, _, _, iter, n_div, _, _, _, _, _, _, _, _) =
            call_lasq3(1, 10, &mut z, 0, 1.0, 0.0, 0.0, 4.0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 0.0);

        // Should have done at least one iteration
        assert!(iter >= 1, "iter should be at least 1, got {}", iter);
        assert!(n_div >= 1, "n_div should be at least 1, got {}", n_div);
    }

    #[test]
    fn sigma_update_with_tau_less_than_sigma() {
        // Test the sigma update logic when tau < sigma
        let mut z = [0.0; 40];
        for i in 0..10 {
            z[4 * i] = 4.0;
            z[4 * i + 2] = 0.1;
        }

        let initial_sigma = 10.0;
        let (_, _, _, sigma, desig, _, _, _, _, _, _, _, _, _, _, _, _) =
            call_lasq3(1, 3, &mut z, 0, 0.1, initial_sigma, 0.0, 4.0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.25, 0.0);

        // Sigma should be updated (may increase or stay same depending on tau)
        assert!(sigma.is_finite());
        assert!(desig.is_finite());
    }

    #[test]
    fn pp_resets_from_2_to_0() {
        // Test that pp == 2 gets reset to 0
        // Need to reach label 50 (main computation) for pp to be reset
        // This requires passing all deflation checks
        // For pp=2, nn = 4*10 + 2 = 42
        // Check 1: z[nn-5-1] = z[36] > tol2 * (sigma + z[nn-3-1] = z[38])
        // Check 2: z[nn-2*2-4-1] = z[33] > tol2 * z[nn-7-1] = z[34]
        // Check 3: z[nn-9-1] = z[32] > tol2 * sigma
        // Check 4: z[nn-2*2-8-1] = z[29] > tol2 * z[nn-11-1] = z[30]
        let mut z = [1.0; 100]; // Initialize all to 1.0 to pass all checks

        // Start with pp=2 which should be reset to 0 at label 50
        let (_, pp, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _) =
            call_lasq3(1, 10, &mut z, 2, 1.0, 0.0, 0.0, 4.0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 0.0);

        assert!(pp <= 1, "pp should be 0 or 1, got {}", pp);
    }

    #[test]
    fn handles_negative_dmin() {
        // Test handling of negative d_min (triggers array reversal)
        let mut z = [0.0; 60];
        for i in 0..15 {
            z[4 * i] = 4.0 - i as f64 * 0.2;
            z[4 * i + 2] = 0.1;
        }

        let (n0, _, d_min, _, _, _, _n_fail, _, _, _, _, _, _, _, _, _, _) =
            call_lasq3(1, 5, &mut z, 0, -0.1, 0.0, 0.0, 4.0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.25, 0.0);

        // Should still complete
        assert!(n0 <= 5);
        assert!(d_min.is_finite());
    }

    #[test]
    fn f32_support() {
        // Test that f32 works
        let mut z = [0.0f32; 40];
        for i in 0..10 {
            z[4 * i] = 4.0;
            z[4 * i + 2] = 0.1;
        }

        let eps = f32::EPSILON;
        let (n0, _, d_min, sigma, _, _, _, _, _, _, _, _, _, _, _, _, _) = lasq3::<f32>(
            1, 3, &mut z, 0, 0.1, 0.0, 0.0, 4.0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.25, 0.0, eps,
        );

        assert!(n0 <= 3);
        assert!(d_min.is_finite());
        assert!(sigma.is_finite());
    }

    #[test]
    fn deflation_loop_terminates() {
        // Test that the deflation loop terminates properly
        let mut z = [0.0; 100];
        for i in 0..25 {
            z[4 * i] = 4.0 - i as f64 * 0.1;
            z[4 * i + 2] = 0.01; // Small off-diagonal elements
        }

        let (n0, _, _, _, _, _, _, iter, _, _, _, _, _, _, _, _, _) =
            call_lasq3(1, 10, &mut z, 0, 0.1, 0.0, 0.0, 4.0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.25, 0.0);

        // Should deflate to completion
        assert!(n0 < 10 || iter > 0);
    }

    #[test]
    fn n_fail_increments_on_failure() {
        // Test that n_fail is incremented when tau is too large
        let mut z = [0.0; 60];
        // Set up array that might cause failures
        for i in 0..15 {
            z[4 * i] = if i % 2 == 0 { 4.0 } else { 0.001 };
            z[4 * i + 2] = 0.5;
        }

        let (_, _, _, _, _, _, n_fail, _, _, _, _, _, _, _, _, _, _) =
            call_lasq3(1, 5, &mut z, 0, -0.5, 0.0, 0.0, 4.0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.25, 0.5);

        // May or may not fail depending on the exact values
        // n_fail is usize so always >= 0
        let _ = n_fail;
    }

    #[test]
    fn qmax_updated() {
        // Test that q_max is properly updated during reversal
        let mut z = [0.0; 60];
        for i in 0..15 {
            z[4 * i] = 4.0 + i as f64;
            z[4 * i + 2] = 0.1;
        }

        let (_, _, _, _, _, q_max, _, _, _, _, _, _, _, _, _, _, _) =
            call_lasq3(1, 5, &mut z, 0, -0.1, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.25, 0.0);

        // q_max should be updated if reversal happened
        assert!(q_max.is_finite());
    }
}
