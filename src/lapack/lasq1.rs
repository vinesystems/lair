use std::ops::{AddAssign, MulAssign};

use super::las2::las2;
use super::lascl;
use super::lasq2::lasq2;
use crate::Real;

/// Computes the singular values of a real N-by-N bidiagonal matrix with
/// diagonal D and off-diagonal E. The singular values are computed to
/// high relative accuracy.
///
/// This is a Rust translation of LAPACK's DLASQ1.
///
/// # Arguments
///
/// * `d` - On entry, the diagonal elements of the bidiagonal matrix.
///   On exit, if successful, the singular values in decreasing order.
/// * `e` - On entry, the off-diagonal elements of the bidiagonal matrix (length n-1).
///   On exit, the contents are overwritten.
/// * `work` - Workspace array of dimension at least 4*n.
///
/// # Returns
///
/// * `Ok(())` on successful completion.
/// * `Err(info)` where `info > 0` indicates an algorithm failure:
///   - `info = 1`: Split not handled properly in lasq2
///   - `info = 2`: Maximum number of iterations exceeded in lasq2
///   - `info = 3`: Failed to converge in lasq2
///
/// # Notes
///
/// This routine uses the dqds algorithm (Differential Quotient-Difference with Shifts)
/// to compute singular values of a bidiagonal matrix. The algorithm was originally
/// presented by Fernando and Parlett (1994).
#[allow(dead_code)]
#[allow(clippy::many_single_char_names, clippy::similar_names)]
pub(crate) fn lasq1<A>(d: &mut [A], e: &mut [A], work: &mut [A]) -> Result<(), isize>
where
    A: Real + AddAssign + MulAssign,
{
    let n = d.len();

    // Quick return if possible
    if n == 0 {
        return Ok(());
    }

    if n == 1 {
        d[0] = d[0].abs();
        return Ok(());
    }

    if n == 2 {
        let (sigmn, sigmx) = las2(d[0], e[0], d[1]);
        d[0] = sigmx;
        d[1] = sigmn;
        return Ok(());
    }

    let zero = A::zero();
    let one = A::one();
    let two = one + one;

    // Estimate the largest singular value
    let mut sigmx = zero;
    for i in 0..n - 1 {
        d[i] = d[i].abs();
        sigmx = sigmx.max(e[i].abs());
    }
    d[n - 1] = d[n - 1].abs();

    // Early return if SIGMX is zero (matrix is already diagonal)
    if sigmx == zero {
        d.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        return Ok(());
    }

    for val in d.iter().take(n) {
        sigmx = sigmx.max(*val);
    }

    // Copy D and E into WORK (in the Z format) and scale (squaring the
    // input data makes scaling by a power of the radix pointless).
    let eps = A::epsilon() / two; // DLAMCH('Precision')
    let safmin = A::min_positive_value(); // DLAMCH('Safe minimum')
    let scale = (eps / safmin).sqrt();

    // DCOPY( N, D, 1, WORK( 1 ), 2 )
    for i in 0..n {
        work[2 * i] = d[i];
    }
    // DCOPY( N-1, E, 1, WORK( 2 ), 2 )
    for i in 0..n - 1 {
        work[2 * i + 1] = e[i];
    }

    // DLASCL( 'G', 0, 0, SIGMX, SCALE, 2*N-1, 1, WORK, 2*N-1, IINFO )
    // Scale WORK by SCALE/SIGMX using safe scaling to avoid overflow
    lascl::general(sigmx, scale, &mut work[..2 * n - 1]);

    // Compute the q's and e's by squaring
    for w in work.iter_mut().take(2 * n - 1) {
        *w *= *w;
    }
    work[2 * n - 1] = zero;

    // Call DLASQ2 to compute singular values
    let result = lasq2(n, work);

    match result {
        Ok(()) => {
            // Success: take square roots and unscale
            for (i, d_val) in d.iter_mut().enumerate().take(n) {
                *d_val = work[i].sqrt();
            }
            // DLASCL( 'G', 0, 0, SCALE, SIGMX, N, 1, D, N, IINFO )
            // Unscale D by SIGMX/SCALE using safe scaling
            lascl::general(scale, sigmx, &mut d[..n]);
            Ok(())
        }
        Err(2) => {
            // Maximum number of iterations exceeded.
            // Move data from WORK into D and E so the calling subroutine can try to finish.
            for i in 0..n {
                d[i] = work[2 * i].sqrt();
                e[i] = work[2 * i + 1].sqrt();
            }
            // Unscale D and E using safe scaling
            lascl::general(scale, sigmx, &mut d[..n]);
            lascl::general(scale, sigmx, &mut e[..n]);
            Err(2)
        }
        Err(info) => Err(info),
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn test_empty() {
        let mut d: [f64; 0] = [];
        let mut e: [f64; 0] = [];
        let mut work: [f64; 0] = [];
        assert!(lasq1(&mut d, &mut e, &mut work).is_ok());
    }

    #[test]
    fn test_single_element() {
        let mut d = [3.0f64];
        let mut e: [f64; 0] = [];
        let mut work = [0.0f64; 4];
        assert!(lasq1(&mut d, &mut e, &mut work).is_ok());
        assert_eq!(d[0], 3.0);
    }

    #[test]
    fn test_single_element_negative() {
        let mut d = [-5.0f64];
        let mut e: [f64; 0] = [];
        let mut work = [0.0f64; 4];
        assert!(lasq1(&mut d, &mut e, &mut work).is_ok());
        assert_eq!(d[0], 5.0); // Should take absolute value
    }

    #[test]
    fn test_two_by_two() {
        // Simple 2x2 bidiagonal matrix:
        // [3 1]
        // [0 2]
        // Singular values can be computed via las2
        let mut d = [3.0f64, 2.0];
        let mut e = [1.0f64];
        let mut work = [0.0f64; 8];
        assert!(lasq1(&mut d, &mut e, &mut work).is_ok());
        // Singular values should be in descending order
        assert!(d[0] >= d[1]);
        // Check they are positive
        assert!(d[0] > 0.0);
        assert!(d[1] > 0.0);
    }

    #[test]
    fn test_diagonal_matrix() {
        // If all off-diagonal elements are zero, singular values are |d_i| sorted
        let mut d = [1.0f64, 4.0, 2.0, 3.0];
        let mut e = [0.0f64, 0.0, 0.0];
        let mut work = [0.0f64; 16];
        assert!(lasq1(&mut d, &mut e, &mut work).is_ok());
        // Should be sorted in descending order
        assert_abs_diff_eq!(d[0], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[1], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[2], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[3], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diagonal_with_negative_entries() {
        // Diagonal entries with negative values - should return absolute values sorted
        let mut d = [-1.0f64, -4.0, 2.0, -3.0];
        let mut e = [0.0f64, 0.0, 0.0];
        let mut work = [0.0f64; 16];
        assert!(lasq1(&mut d, &mut e, &mut work).is_ok());
        // Should be sorted in descending order of absolute values
        assert_abs_diff_eq!(d[0], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[1], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[2], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[3], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_f32_support() {
        let mut d = [3.0f32];
        let mut e: [f32; 0] = [];
        let mut work = [0.0f32; 4];
        assert!(lasq1(&mut d, &mut e, &mut work).is_ok());
        assert_eq!(d[0], 3.0);
    }

    #[test]
    fn test_simple_bidiagonal() {
        // A simple 3x3 bidiagonal matrix
        // [2 1 0]
        // [0 3 1]
        // [0 0 4]
        let mut d = [2.0f64, 3.0, 4.0];
        let mut e = [1.0f64, 1.0];
        let mut work = [0.0f64; 12];
        let result = lasq1(&mut d, &mut e, &mut work);
        assert!(result.is_ok(), "Expected convergence, got {:?}", result);

        // Check singular values are positive and sorted
        assert!(d[0] > 0.0);
        assert!(d[1] > 0.0);
        assert!(d[2] > 0.0);
        assert!(d[0] >= d[1]);
        assert!(d[1] >= d[2]);
    }

    #[test]
    fn test_identity_like() {
        // Bidiagonal with all 1s on diagonal, small off-diagonal
        let mut d = [1.0f64, 1.0, 1.0, 1.0];
        let mut e = [0.1f64, 0.1, 0.1];
        let mut work = [0.0f64; 16];
        let result = lasq1(&mut d, &mut e, &mut work);
        assert!(result.is_ok(), "Expected convergence, got {:?}", result);

        // Check singular values are positive and sorted
        for i in 0..4 {
            assert!(d[i] > 0.0, "Singular value {} should be positive", i);
            if i > 0 {
                assert!(d[i - 1] >= d[i], "Singular values should be sorted");
            }
        }
    }
}
