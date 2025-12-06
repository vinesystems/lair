use crate::Real;

/// Computes the singular value decomposition of a 2-by-2 triangular matrix.
///
/// LASV2 computes the singular value decomposition of a 2-by-2
/// triangular matrix
///    [  F   G  ]
///    [  0   H  ].
///
/// On return, `ssmax.abs()` is the larger singular value, `ssmin.abs()` is the
/// smaller singular value, and `(csl, snl)` and `(csr, snr)` are the left and
/// right singular vectors for `ssmax.abs()`, giving the decomposition
///
///    [ CSL  SNL ] [  F   G  ] [ CSR -SNR ]  =  [ SSMAX   0   ]
///    [-SNL  CSL ] [  0   H  ] [ SNR  CSR ]     [  0    SSMIN ].
///
/// # Arguments
///
/// * `f` - The (1,1) element of the 2-by-2 matrix.
/// * `g` - The (1,2) element of the 2-by-2 matrix.
/// * `h` - The (2,2) element of the 2-by-2 matrix.
///
/// # Returns
///
/// A tuple `(ssmin, ssmax, snr, csr, snl, csl)` where:
/// * `ssmin` - The smaller singular value (may be negative).
/// * `ssmax` - The larger singular value (may be negative).
/// * `snr` - The sine component of the right singular vector.
/// * `csr` - The cosine component of the right singular vector.
/// * `snl` - The sine component of the left singular vector.
/// * `csl` - The cosine component of the left singular vector.
#[allow(dead_code)]
#[allow(clippy::many_single_char_names, clippy::similar_names)]
pub(crate) fn lasv2<T: Real>(f: T, g: T, h: T) -> (T, T, T, T, T, T) {
    let zero = T::zero();
    let half = T::from(0.5).unwrap();
    let one = T::one();
    let two = T::from(2.0).unwrap();
    let four = T::from(4.0).unwrap();

    let mut ft = f;
    let fa = ft.abs();
    let mut ht = h;
    let mut ha = h.abs();

    // PMAX points to the maximum absolute element of matrix
    //   PMAX = 1 if F largest in absolute values
    //   PMAX = 2 if G largest in absolute values
    //   PMAX = 3 if H largest in absolute values
    let mut pmax = 1;
    let swap = ha > fa;
    if swap {
        pmax = 3;
        std::mem::swap(&mut ft, &mut ht);
        ha = fa;
        // Now FA >= HA (conceptually, fa is now the absolute value of ft)
    }
    let fa = ft.abs();

    let gt = g;
    let ga = gt.abs();

    let (mut ssmin, mut ssmax, clt, slt, crt, srt): (T, T, T, T, T, T) = if ga == zero {
        // Diagonal matrix
        (ha, fa, one, zero, one, zero)
    } else if ga > fa && (fa / ga) < T::eps() {
        // Case of very large GA
        pmax = 2;
        let ssmax = ga;
        let ssmin = if ha > one {
            fa / (ga / ha)
        } else {
            (fa / ga) * ha
        };
        let clt = one;
        let slt = ht / gt;
        let srt = one;
        let crt = ft / gt;
        (ssmin, ssmax, clt, slt, crt, srt)
    } else {
        // Normal case
        if ga > fa {
            pmax = 2;
        }

        let d = fa - ha;
        let l = if d == fa {
            // Copes with infinite F or H
            one
        } else {
            d / fa
        };

        // Note that 0 <= L <= 1
        let m = gt / ft;

        // Note that abs(M) <= 1/macheps
        let t = two - l;

        // Note that T >= 1
        let mm = m * m;
        let tt = t * t;
        let s = (tt + mm).sqrt();

        // Note that 1 <= S <= 1 + 1/macheps
        let r = if l == zero {
            m.abs()
        } else {
            (l * l + mm).sqrt()
        };

        // Note that 0 <= R <= 1 + 1/macheps
        let a = half * (s + r);

        // Note that 1 <= A <= 1 + abs(M)
        let ssmin = ha / a;
        let ssmax = fa * a;

        let t = if mm == zero {
            // Note that M is very tiny
            if l == zero {
                sign(two, ft) * sign(one, gt)
            } else {
                gt / sign(d, ft) + m / t
            }
        } else {
            (m / (s + t) + m / (r + l)) * (one + a)
        };

        let l = (t * t + four).sqrt();
        let crt = two / l;
        let srt = t / l;
        let clt = (crt + srt * m) / a;
        let slt = (ht / ft) * srt / a;

        (ssmin, ssmax, clt, slt, crt, srt)
    };

    let (csl, snl, csr, snr) = if swap {
        (srt, crt, slt, clt)
    } else {
        (clt, slt, crt, srt)
    };

    // Correct signs of SSMAX and SSMIN
    let tsign = match pmax {
        1 => sign(one, csr) * sign(one, csl) * sign(one, f),
        2 => sign(one, snr) * sign(one, csl) * sign(one, g),
        _ => sign(one, snr) * sign(one, snl) * sign(one, h),
    };
    ssmax = sign(ssmax, tsign);
    ssmin = sign(ssmin, tsign * sign(one, f) * sign(one, h));

    (ssmin, ssmax, snr, csr, snl, csl)
}

/// Returns the absolute value of `a` with the sign of `b`.
/// Equivalent to Fortran's SIGN intrinsic.
#[inline]
fn sign<T: Real>(a: T, b: T) -> T {
    if b >= T::zero() {
        a.abs()
    } else {
        -a.abs()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[allow(clippy::too_many_arguments)]
    /// Verifies the SVD decomposition:
    /// [ CSL  SNL ] [  F   G  ] [ CSR -SNR ]  =  [ SSMAX   0   ]
    /// [-SNL  CSL ] [  0   H  ] [ SNR  CSR ]     [  0    SSMIN ]
    fn verify_svd(
        f: f64,
        g: f64,
        h: f64,
        ssmin: f64,
        ssmax: f64,
        snr: f64,
        csr: f64,
        snl: f64,
        csl: f64,
        tol: f64,
    ) {
        // Left singular matrix U = [ CSL  SNL; -SNL  CSL ]
        // Right singular matrix V = [ CSR -SNR; SNR  CSR ]
        // Original matrix A = [ F  G; 0  H ]
        // Singular values S = [ SSMAX  0; 0  SSMIN ]
        // We verify U * A * V = S

        // Compute U * A
        // U * A = [ CSL  SNL ] [ F  G ]
        //         [-SNL  CSL ] [ 0  H ]
        let ua_11 = csl * f;
        let ua_12 = csl * g + snl * h;
        let ua_21 = -snl * f;
        let ua_22 = -snl * g + csl * h;

        // Compute (U * A) * V
        // (U * A) * V = [ ua_11  ua_12 ] [ CSR -SNR ]
        //               [ ua_21  ua_22 ] [ SNR  CSR ]
        let s_11 = ua_11 * csr + ua_12 * snr;
        let s_12 = -ua_11 * snr + ua_12 * csr;
        let s_21 = ua_21 * csr + ua_22 * snr;
        let s_22 = -ua_21 * snr + ua_22 * csr;

        // Verify the result equals the singular value matrix
        assert_abs_diff_eq!(s_11, ssmax, epsilon = tol);
        assert_abs_diff_eq!(s_12, 0.0, epsilon = tol);
        assert_abs_diff_eq!(s_21, 0.0, epsilon = tol);
        assert_abs_diff_eq!(s_22, ssmin, epsilon = tol);

        // Verify singular vectors are unit vectors
        assert_abs_diff_eq!(csl * csl + snl * snl, 1.0, epsilon = tol);
        assert_abs_diff_eq!(csr * csr + snr * snr, 1.0, epsilon = tol);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn lasv2_diagonal() {
        // Diagonal matrix: G = 0, |F| > |H|
        let (ssmin, ssmax, snr, csr, snl, csl) = lasv2(4.0_f64, 0.0, 3.0);
        assert_eq!(ssmin, 3.0);
        assert_eq!(ssmax, 4.0);
        assert_eq!(snr, 0.0);
        assert_eq!(csr, 1.0);
        assert_eq!(snl, 0.0);
        assert_eq!(csl, 1.0);
        verify_svd(4.0, 0.0, 3.0, ssmin, ssmax, snr, csr, snl, csl, 1e-10);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn lasv2_diagonal_swap() {
        // Diagonal matrix with |H| > |F|
        let (ssmin, ssmax, snr, csr, snl, csl) = lasv2(2.0_f64, 0.0, 5.0);
        assert_eq!(ssmin.abs(), 2.0);
        assert_eq!(ssmax.abs(), 5.0);
        verify_svd(2.0, 0.0, 5.0, ssmin, ssmax, snr, csr, snl, csl, 1e-10);
    }

    #[test]
    fn lasv2_upper_triangular() {
        // Upper triangular matrix
        let (ssmin, ssmax, snr, csr, snl, csl) = lasv2(3.0_f64, 4.0, 5.0);
        // Verify the decomposition
        verify_svd(3.0, 4.0, 5.0, ssmin, ssmax, snr, csr, snl, csl, 1e-10);
        // Verify singular values are correct magnitude
        assert!(ssmax.abs() >= ssmin.abs());
    }

    #[test]
    fn lasv2_negative_elements() {
        // Matrix with negative elements
        let (ssmin, ssmax, snr, csr, snl, csl) = lasv2(-3.0_f64, 4.0, -5.0);
        verify_svd(-3.0, 4.0, -5.0, ssmin, ssmax, snr, csr, snl, csl, 1e-10);
    }

    #[test]
    fn lasv2_all_negative() {
        // Matrix with all negative elements
        let (ssmin, ssmax, snr, csr, snl, csl) = lasv2(-3.0_f64, -4.0, -5.0);
        verify_svd(-3.0, -4.0, -5.0, ssmin, ssmax, snr, csr, snl, csl, 1e-10);
    }

    #[test]
    fn lasv2_zero_f() {
        // F = 0
        let (ssmin, ssmax, snr, csr, snl, csl) = lasv2(0.0_f64, 4.0, 5.0);
        verify_svd(0.0, 4.0, 5.0, ssmin, ssmax, snr, csr, snl, csl, 1e-10);
    }

    #[test]
    fn lasv2_zero_h() {
        // H = 0
        let (ssmin, ssmax, snr, csr, snl, csl) = lasv2(3.0_f64, 4.0, 0.0);
        verify_svd(3.0, 4.0, 0.0, ssmin, ssmax, snr, csr, snl, csl, 1e-10);
    }

    #[test]
    fn lasv2_large_g() {
        // Very large G compared to F and H
        let (ssmin, ssmax, snr, csr, snl, csl) = lasv2(1.0_f64, 1e10, 1.0);
        verify_svd(1.0, 1e10, 1.0, ssmin, ssmax, snr, csr, snl, csl, 1e-5);
    }

    #[test]
    fn lasv2_small_values() {
        // Small values
        let (ssmin, ssmax, snr, csr, snl, csl) = lasv2(1e-10_f64, 2e-10, 3e-10);
        verify_svd(1e-10, 2e-10, 3e-10, ssmin, ssmax, snr, csr, snl, csl, 1e-10);
    }

    #[test]
    fn lasv2_f32() {
        // Test with f32
        let (ssmin, ssmax, snr, csr, snl, csl) = lasv2(3.0_f32, 4.0, 5.0);
        // Verify singular vectors are unit vectors
        assert_abs_diff_eq!(csl * csl + snl * snl, 1.0_f32, epsilon = 1e-5);
        assert_abs_diff_eq!(csr * csr + snr * snr, 1.0_f32, epsilon = 1e-5);
        assert!(ssmax.abs() >= ssmin.abs());
    }

    #[test]
    fn lasv2_identity_like() {
        // Near-identity matrix [1 0; 0 1]
        let (ssmin, ssmax, snr, csr, snl, csl) = lasv2(1.0_f64, 0.0, 1.0);
        verify_svd(1.0, 0.0, 1.0, ssmin, ssmax, snr, csr, snl, csl, 1e-10);
    }

    #[test]
    fn lasv2_symmetric_like() {
        // Test with F = H
        let (ssmin, ssmax, snr, csr, snl, csl) = lasv2(5.0_f64, 3.0, 5.0);
        verify_svd(5.0, 3.0, 5.0, ssmin, ssmax, snr, csr, snl, csl, 1e-10);
    }
}
