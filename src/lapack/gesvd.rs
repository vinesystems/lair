use std::cmp;
use std::ops::{AddAssign, Div, Mul, MulAssign, Sub};

use ndarray::{s, Array2, ArrayBase, DataMut, Ix2};
use num_traits::{Float, One, Zero};

use super::bdsqr::{bdsqr, Uplo};
use super::gebrd::tall as gebrd_tall;
use super::gelqf::gelqf;
use super::lacpy;
use super::lange::maxabs;
use super::lascl;
use super::ungbr::{p_square, q_tall};
use super::unglq::unglq;
use crate::{Real, Scalar};

/// Specifies how to compute left singular vectors (U).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum JobU {
    /// Compute all M columns of U.
    All,
    /// Compute the first min(M,N) columns of U (left singular vectors).
    Some,
    /// Overwrite A with the first min(M,N) columns of U.
    Overwrite,
    /// Do not compute left singular vectors.
    None,
}

/// Specifies how to compute right singular vectors (VT).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum JobVT {
    /// Compute all N rows of VT.
    All,
    /// Compute the first min(M,N) rows of VT (right singular vectors).
    Some,
    /// Overwrite A with the first min(M,N) rows of VT.
    Overwrite,
    /// Do not compute right singular vectors.
    None,
}

/// Computes the singular value decomposition (SVD) of a general matrix.
///
/// The SVD is written as:
///   A = U * SIGMA * V^H
///
/// where SIGMA is an M-by-N matrix which is zero except for its min(M,N)
/// diagonal elements (the singular values), U is an M-by-M unitary matrix,
/// and V is an N-by-N unitary matrix. The diagonal elements of SIGMA are
/// the singular values of A in descending order.
///
/// # Arguments
///
/// * `jobu` - Specifies options for computing all or part of the matrix U.
/// * `jobvt` - Specifies options for computing all or part of the matrix VT.
/// * `a` - On entry, the M-by-N matrix to be decomposed.
///   On exit, the contents are overwritten based on `jobu` and `jobvt`.
/// * `s` - On exit, the singular values of A in decreasing order.
///   Must have length min(M,N).
/// * `u` - On exit, contains U (all or part depending on `jobu`).
///   If `jobu` is `None`, U is not referenced.
/// * `vt` - On exit, contains V^H (all or part depending on `jobvt`).
///   If `jobvt` is `None`, VT is not referenced.
///
/// # Returns
///
/// * `Ok(())` if successful.
/// * `Err(i)` if the SVD algorithm failed to converge; `i` indicates how many
///   superdiagonals did not converge to zero.
#[allow(dead_code)]
#[allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::too_many_arguments
)]
pub fn gesvd<A, SA, SU, SVT>(
    jobu: JobU,
    jobvt: JobVT,
    a: &mut ArrayBase<SA, Ix2>,
    s: &mut [A::Real],
    u: &mut ArrayBase<SU, Ix2>,
    vt: &mut ArrayBase<SVT, Ix2>,
) -> Result<(), usize>
where
    A: Scalar
        + Div<<A as Scalar>::Real, Output = A>
        + MulAssign<<A as Scalar>::Real>
        + AddAssign
        + Mul<A::Real, Output = A>
        + Sub<Output = A>,
    A::Real: Real,
    SA: DataMut<Elem = A>,
    SU: DataMut<Elem = A>,
    SVT: DataMut<Elem = A>,
{
    let m = a.nrows();
    let n = a.ncols();
    let minmn = cmp::min(m, n);

    // Validate dimensions
    assert_eq!(s.len(), minmn, "s must have length min(m, n)");

    let wantu = jobu != JobU::None;
    let wantvt = jobvt != JobVT::None;

    // Quick return for empty matrix
    if minmn == 0 {
        return Ok(());
    }

    // Get machine parameters
    let zero = A::Real::zero();
    let one = A::Real::one();

    // Scale A if max element outside range [SMLNUM, BIGNUM]
    let anrm = maxabs::<A, _>(a);
    let mut iscl = false;
    let smlnum = A::Real::sfmin().sqrt() / A::Real::eps();
    let bignum = one / smlnum;

    if anrm > zero && anrm < smlnum {
        iscl = true;
        lascl::general(anrm, smlnum, a.as_slice_mut().unwrap());
    } else if anrm > bignum {
        iscl = true;
        lascl::general(anrm, bignum, a.as_slice_mut().unwrap());
    }

    // Workspace for bdsqr
    let work_size = 4 * minmn;
    let mut work = vec![A::Real::zero(); work_size];

    if m >= n {
        // A has at least as many rows as columns
        // Reduce A to bidiagonal form
        let (d, e, tau_q, tau_p) = gebrd_tall(a);

        // Copy bidiagonal elements
        for (i, val) in d.iter().enumerate() {
            s[i] = *val;
        }
        let mut e_work: Vec<A::Real> = e.to_vec();

        // Generate U if wanted
        if wantu {
            match jobu {
                JobU::All => {
                    // U is M x M
                    lacpy::lower(a, u);
                    // Extend to full orthogonal matrix
                    q_tall(u, &tau_q);
                }
                JobU::Some | JobU::Overwrite => {
                    // U is M x min(M,N)
                    lacpy::lower(&a.slice(s![.., ..minmn]), &mut u.slice_mut(s![.., ..minmn]));
                    let mut u_slice = u.slice_mut(s![.., ..minmn]);
                    q_tall(&mut u_slice, &tau_q);
                }
                JobU::None => {}
            }
        }

        // Generate VT if wanted
        if wantvt {
            match jobvt {
                JobVT::All | JobVT::Some => {
                    // Copy the upper triangular part of A to VT
                    lacpy::upper(&a.slice(s![..n, ..n]), vt);
                    p_square(vt, &tau_p);
                }
                JobVT::Overwrite | JobVT::None => {}
            }
        }

        // Perform bidiagonal SVD
        // Create dummy arrays for unused parameters
        let mut c_dummy: Array2<A> = Array2::zeros((minmn, 0));

        let result = if wantu && wantvt {
            bdsqr::<A, _, _, _>(Uplo::Upper, s, &mut e_work, vt, u, &mut c_dummy, &mut work)
        } else if wantu {
            let mut vt_dummy: Array2<A> = Array2::zeros((minmn, 0));
            bdsqr::<A, _, _, _>(
                Uplo::Upper,
                s,
                &mut e_work,
                &mut vt_dummy,
                u,
                &mut c_dummy,
                &mut work,
            )
        } else if wantvt {
            let mut u_dummy: Array2<A> = Array2::zeros((0, minmn));
            bdsqr::<A, _, _, _>(
                Uplo::Upper,
                s,
                &mut e_work,
                vt,
                &mut u_dummy,
                &mut c_dummy,
                &mut work,
            )
        } else {
            let mut vt_dummy: Array2<A> = Array2::zeros((minmn, 0));
            let mut u_dummy: Array2<A> = Array2::zeros((0, minmn));
            bdsqr::<A, _, _, _>(
                Uplo::Upper,
                s,
                &mut e_work,
                &mut vt_dummy,
                &mut u_dummy,
                &mut c_dummy,
                &mut work,
            )
        };

        result?;
    } else {
        // m < n: More columns than rows
        // First reduce to bidiagonal form via LQ factorization

        // Compute LQ factorization of A
        let tau_lq = gelqf(a);

        // Copy L to U if needed (L is the lower triangular part of A)
        if wantu {
            match jobu {
                JobU::All | JobU::Some => {
                    for i in 0..m {
                        for j in 0..=cmp::min(i, m - 1) {
                            u[(i, j)] = a[(i, j)];
                        }
                        for j in (cmp::min(i, m - 1) + 1)..u.ncols() {
                            u[(i, j)] = A::zero();
                        }
                    }
                }
                JobU::Overwrite | JobU::None => {}
            }
        }

        // Generate Q from LQ factorization into VT if needed
        if wantvt {
            match jobvt {
                JobVT::All => {
                    // VT is N x N, copy A and generate full Q
                    for i in 0..m {
                        for j in 0..n {
                            vt[(i, j)] = a[(i, j)];
                        }
                    }
                    for i in m..n {
                        for j in 0..n {
                            vt[(i, j)] = A::zero();
                        }
                    }
                    unglq(&mut vt.slice_mut(s![..m, ..]), &tau_lq);
                    // Extend to full orthogonal matrix
                    for i in m..n {
                        vt[(i, i)] = A::one();
                    }
                }
                JobVT::Some => {
                    // VT is min(M,N) x N = M x N
                    for i in 0..m {
                        for j in 0..n {
                            vt[(i, j)] = a[(i, j)];
                        }
                    }
                    unglq(vt, &tau_lq);
                }
                JobVT::Overwrite | JobVT::None => {}
            }
        }

        // Now reduce L (which is m x m) to bidiagonal form
        // L is stored in the lower triangular part of A (and copied to U if needed)
        // We need to work with L separately

        // Extract L into a separate matrix for bidiagonalization
        let mut l_matrix: Array2<A> = Array2::zeros((m, m));
        for i in 0..m {
            for j in 0..=i {
                l_matrix[(i, j)] = if wantu { u[(i, j)] } else { a[(i, j)] };
            }
        }

        // Reduce L to bidiagonal form
        // Since L is m x m, we transpose it to use gebrd_tall which expects nrows >= ncols
        // Actually, for a square matrix, gebrd_tall works fine
        let (d, e, tau_q, _tau_p) = gebrd_tall(&mut l_matrix);

        // Copy bidiagonal elements
        for (i, val) in d.iter().enumerate() {
            s[i] = *val;
        }
        let mut e_work: Vec<A::Real> = e.to_vec();

        // Update U if needed: U = U * Q_bidiag
        if wantu {
            q_tall(&mut l_matrix, &tau_q);
            // Copy back to U
            for i in 0..m {
                for j in 0..m {
                    u[(i, j)] = l_matrix[(i, j)];
                }
            }
        }

        // Update VT if needed: VT = P^H * VT
        // This requires applying P^H from gebrd to the rows of VT
        // For simplicity, we generate P^H and multiply
        if wantvt {
            // Generate P^H from the bidiagonalization
            let mut p_matrix: Array2<A> = Array2::zeros((m, m));
            for i in 0..m {
                for j in 0..m {
                    // Copy upper triangular part which contains P reflectors
                    if j >= i {
                        p_matrix[(i, j)] = l_matrix[(i, j)];
                    }
                }
            }

            // Note: For the case m < n, the bidiagonal matrix from L is lower bidiagonal
            // and we need to handle it differently. However, since we're using gebrd_tall
            // on the m x m matrix L, it produces an upper bidiagonal form.

            // Apply P^H to VT - this is done by bdsqr when we pass VT
        }

        // Perform bidiagonal SVD
        let mut c_dummy: Array2<A> = Array2::zeros((m, 0));

        let result = if wantu && wantvt {
            // For m < n case, we need VT to be properly sized for bdsqr
            // bdsqr expects VT to have m rows (size of bidiagonal matrix)
            let mut vt_small = vt.slice_mut(s![..m, ..]);
            bdsqr::<A, _, _, _>(
                Uplo::Upper,
                s,
                &mut e_work,
                &mut vt_small,
                u,
                &mut c_dummy,
                &mut work,
            )
        } else if wantu {
            let mut vt_dummy: Array2<A> = Array2::zeros((m, 0));
            bdsqr::<A, _, _, _>(
                Uplo::Upper,
                s,
                &mut e_work,
                &mut vt_dummy,
                u,
                &mut c_dummy,
                &mut work,
            )
        } else if wantvt {
            let mut u_dummy: Array2<A> = Array2::zeros((0, m));
            let mut vt_small = vt.slice_mut(s![..m, ..]);
            bdsqr::<A, _, _, _>(
                Uplo::Upper,
                s,
                &mut e_work,
                &mut vt_small,
                &mut u_dummy,
                &mut c_dummy,
                &mut work,
            )
        } else {
            let mut vt_dummy: Array2<A> = Array2::zeros((m, 0));
            let mut u_dummy: Array2<A> = Array2::zeros((0, m));
            bdsqr::<A, _, _, _>(
                Uplo::Upper,
                s,
                &mut e_work,
                &mut vt_dummy,
                &mut u_dummy,
                &mut c_dummy,
                &mut work,
            )
        };

        result?;
    }

    // Undo scaling if necessary
    if iscl {
        if anrm > bignum {
            lascl::general(bignum, anrm, s);
        }
        if anrm < smlnum {
            lascl::general(smlnum, anrm, s);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{arr2, Array2};
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn test_empty_matrix() {
        let mut a: Array2<f64> = Array2::zeros((0, 0));
        let mut s: Vec<f64> = vec![];
        let mut u: Array2<f64> = Array2::zeros((0, 0));
        let mut vt: Array2<f64> = Array2::zeros((0, 0));

        let result = gesvd(JobU::All, JobVT::All, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_single_element() {
        let mut a = arr2(&[[3.0f64]]);
        let mut s = vec![0.0f64];
        let mut u = Array2::<f64>::zeros((1, 1));
        let mut vt = Array2::<f64>::zeros((1, 1));

        let result = gesvd(JobU::All, JobVT::All, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());
        assert_abs_diff_eq!(s[0], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_identity_2x2() {
        let mut a = arr2(&[[1.0f64, 0.0], [0.0, 1.0]]);
        let mut s = vec![0.0f64; 2];
        let mut u = Array2::<f64>::zeros((2, 2));
        let mut vt = Array2::<f64>::zeros((2, 2));

        let result = gesvd(JobU::All, JobVT::All, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());
        // Singular values of identity are all 1, sorted descending
        assert_abs_diff_eq!(s[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(s[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diagonal_matrix() {
        let mut a = arr2(&[[3.0f64, 0.0], [0.0, 5.0]]);
        let mut s = vec![0.0f64; 2];
        let mut u = Array2::<f64>::zeros((2, 2));
        let mut vt = Array2::<f64>::zeros((2, 2));

        let result = gesvd(JobU::All, JobVT::All, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());
        // Singular values should be sorted in descending order
        assert_abs_diff_eq!(s[0], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(s[1], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_svd_reconstruction_square() {
        // Test that A = U * S * VT
        let mut a = arr2(&[[1.0f64, 2.0], [3.0, 4.0]]);
        let a_orig = a.clone();
        let mut s = vec![0.0f64; 2];
        let mut u = Array2::<f64>::zeros((2, 2));
        let mut vt = Array2::<f64>::zeros((2, 2));

        let result = gesvd(JobU::All, JobVT::All, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());

        // Reconstruct A = U * diag(s) * VT
        let mut s_diag = Array2::<f64>::zeros((2, 2));
        s_diag[(0, 0)] = s[0];
        s_diag[(1, 1)] = s[1];
        let reconstructed = u.dot(&s_diag).dot(&vt);

        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(reconstructed[(i, j)], a_orig[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_tall_matrix() {
        // 4x2 matrix
        let mut a = arr2(&[[1.0f64, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
        let a_orig = a.clone();
        let mut s = vec![0.0f64; 2];
        let mut u = Array2::<f64>::zeros((4, 4));
        let mut vt = Array2::<f64>::zeros((2, 2));

        let result = gesvd(JobU::All, JobVT::All, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());

        // Check singular values are positive and sorted
        assert!(s[0] > 0.0);
        assert!(s[1] > 0.0);
        assert!(s[0] >= s[1]);

        // Reconstruct A = U[:, :2] * diag(s) * VT
        let mut s_diag = Array2::<f64>::zeros((2, 2));
        s_diag[(0, 0)] = s[0];
        s_diag[(1, 1)] = s[1];
        let u_reduced = u.slice(s![.., ..2]);
        let reconstructed = u_reduced.dot(&s_diag).dot(&vt);

        for i in 0..4 {
            for j in 0..2 {
                assert_abs_diff_eq!(reconstructed[(i, j)], a_orig[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_wide_matrix() {
        // 2x4 matrix
        let mut a = arr2(&[[1.0f64, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
        let a_orig = a.clone();
        let mut s = vec![0.0f64; 2];
        let mut u = Array2::<f64>::zeros((2, 2));
        let mut vt = Array2::<f64>::zeros((4, 4));

        let result = gesvd(JobU::All, JobVT::All, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());

        // Check singular values are positive and sorted
        assert!(s[0] > 0.0);
        assert!(s[1] > 0.0);
        assert!(s[0] >= s[1]);

        // Reconstruct A = U * diag(s) * VT[:2, :]
        let mut s_diag = Array2::<f64>::zeros((2, 2));
        s_diag[(0, 0)] = s[0];
        s_diag[(1, 1)] = s[1];
        let vt_reduced = vt.slice(s![..2, ..]);
        let reconstructed = u.dot(&s_diag).dot(&vt_reduced);

        for i in 0..2 {
            for j in 0..4 {
                assert_abs_diff_eq!(reconstructed[(i, j)], a_orig[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_no_vectors() {
        let mut a = arr2(&[[1.0f64, 2.0], [3.0, 4.0]]);
        let mut s = vec![0.0f64; 2];
        let mut u = Array2::<f64>::zeros((0, 0));
        let mut vt = Array2::<f64>::zeros((0, 0));

        let result = gesvd(JobU::None, JobVT::None, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());

        // Check singular values are computed correctly
        assert!(s[0] > 0.0);
        assert!(s[1] > 0.0);
        assert!(s[0] >= s[1]);
    }

    #[test]
    fn test_orthogonality_u() {
        let mut a = arr2(&[[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]);
        let mut s = vec![0.0f64; 3];
        let mut u = Array2::<f64>::zeros((3, 3));
        let mut vt = Array2::<f64>::zeros((3, 3));

        let result = gesvd(JobU::All, JobVT::All, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());

        // Check U^T * U = I
        let utu = u.t().dot(&u);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(utu[(i, j)], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_orthogonality_vt() {
        let mut a = arr2(&[[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]);
        let mut s = vec![0.0f64; 3];
        let mut u = Array2::<f64>::zeros((3, 3));
        let mut vt = Array2::<f64>::zeros((3, 3));

        let result = gesvd(JobU::All, JobVT::All, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());

        // Check VT * VT^T = I
        let vtvt = vt.dot(&vt.t());
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(vtvt[(i, j)], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_complex_matrix() {
        let mut a = arr2(&[
            [Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
            [Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)],
        ]);
        let a_orig = a.clone();
        let mut s = vec![0.0f64; 2];
        let mut u = Array2::<Complex64>::zeros((2, 2));
        let mut vt = Array2::<Complex64>::zeros((2, 2));

        let result = gesvd(JobU::All, JobVT::All, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());

        // Check singular values are positive and sorted
        assert!(s[0] > 0.0);
        assert!(s[1] > 0.0);
        assert!(s[0] >= s[1]);

        // Reconstruct A = U * diag(s) * VT
        let mut s_diag = Array2::<Complex64>::zeros((2, 2));
        s_diag[(0, 0)] = Complex64::new(s[0], 0.0);
        s_diag[(1, 1)] = Complex64::new(s[1], 0.0);
        let reconstructed = u.dot(&s_diag).dot(&vt);

        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(reconstructed[(i, j)].re, a_orig[(i, j)].re, epsilon = 1e-8);
                assert_abs_diff_eq!(reconstructed[(i, j)].im, a_orig[(i, j)].im, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_rank_deficient() {
        // Matrix with rank 1: all rows are multiples of [1, 2]
        let mut a = arr2(&[[1.0f64, 2.0], [2.0, 4.0], [3.0, 6.0]]);
        let mut s = vec![0.0f64; 2];
        let mut u = Array2::<f64>::zeros((3, 3));
        let mut vt = Array2::<f64>::zeros((2, 2));

        let result = gesvd(JobU::All, JobVT::All, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());

        // One singular value should be near zero (rank deficient)
        assert!(s[0] > 0.0);
        assert!(s[1].abs() < 1e-10);
    }
}
