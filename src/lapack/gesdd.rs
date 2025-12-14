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

/// Specifies what to compute in the SVD.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Job {
    /// Compute all M columns of U and all N rows of VT.
    All,
    /// Compute the first min(M,N) columns of U and the first min(M,N) rows of VT.
    Some,
    /// If M >= N, compute the first N columns of U (overwriting A) and all N rows of VT.
    /// If M < N, compute all M columns of U and the first M rows of VT (overwriting A).
    Overwrite,
    /// Compute no singular vectors.
    None,
}

/// Computes the singular value decomposition (SVD) of a general matrix using
/// divide-and-conquer.
///
/// The SVD is written as:
///   A = U * SIGMA * V^H
///
/// where SIGMA is an M-by-N matrix which is zero except for its min(M,N)
/// diagonal elements (the singular values), U is an M-by-M unitary matrix,
/// and V is an N-by-N unitary matrix. The diagonal elements of SIGMA are
/// the singular values of A in descending order.
///
/// Note: This implementation uses the same underlying algorithm as `gesvd`
/// (QR iteration via `bdsqr`) rather than a true divide-and-conquer approach.
/// The interface matches the LAPACK `gesdd` routine.
///
/// # Arguments
///
/// * `job` - Specifies options for computing U and VT.
///   - `Job::All`: All M columns of U and all N rows of VT are returned.
///   - `Job::Some`: The first min(M,N) columns of U and first min(M,N) rows of VT
///     are returned.
///   - `Job::Overwrite`: If M >= N, the first N columns of U are overwritten in A
///     and all N rows of VT are returned in VT. If M < N, all M rows of VT are
///     overwritten in A and all M columns of U are returned in U.
///   - `Job::None`: No singular vectors are computed.
/// * `a` - On entry, the M-by-N matrix to be decomposed.
///   On exit, contents depend on `job`:
///   - If `job` is `Overwrite` and M >= N, A is overwritten with the first N columns of U.
///   - If `job` is `Overwrite` and M < N, A is overwritten with the first M rows of VT.
///   - Otherwise, A is destroyed.
/// * `s` - On exit, the singular values of A in decreasing order.
///   Must have length min(M,N).
/// * `u` - On exit, contains U (all or part depending on `job`).
///   Not referenced if `job` is `None` or (`Overwrite` and M >= N).
/// * `vt` - On exit, contains V^H (all or part depending on `job`).
///   Not referenced if `job` is `None` or (`Overwrite` and M < N).
///
/// # Returns
///
/// * `Ok(())` if successful.
/// * `Err(i)` if the SVD algorithm failed to converge; `i` indicates how many
///   superdiagonals did not converge to zero.
///
/// # Errors
///
/// Returns `Err(i)` if the algorithm fails to converge; `i` is the number of
/// superdiagonals that did not converge to zero.
///
/// # Panics
///
/// Panics if the provided buffers do not match the expected dimensions for
/// the chosen `job`, or if `s` is not `min(m, n)` long.
#[allow(dead_code)]
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_lines)]
pub fn gesdd<A, SA, SU, SVT>(
    job: Job,
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

    // Quick return for empty matrix
    if minmn == 0 {
        return Ok(());
    }

    // Check matrix dimensions for u and vt based on job
    match job {
        Job::All => {
            assert!(u.nrows() >= m && u.ncols() >= m, "u must be at least m x m");
            assert!(
                vt.nrows() >= n && vt.ncols() >= n,
                "vt must be at least n x n"
            );
        }
        Job::Some => {
            assert!(
                u.nrows() >= m && u.ncols() >= minmn,
                "u must be at least m x min(m,n)"
            );
            assert!(
                vt.nrows() >= minmn && vt.ncols() >= n,
                "vt must be at least min(m,n) x n"
            );
        }
        Job::Overwrite => {
            if m >= n {
                assert!(
                    vt.nrows() >= n && vt.ncols() >= n,
                    "vt must be at least n x n"
                );
            } else {
                assert!(u.nrows() >= m && u.ncols() >= m, "u must be at least m x m");
            }
        }
        Job::None => {
            // No dimension checks needed for u/vt
        }
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
        gesdd_tall(job, a, s, u, vt, &mut work)?;
    } else {
        // m < n: More columns than rows
        gesdd_wide(job, a, s, u, vt, &mut work)?;
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

/// Handle the case when m >= n (tall or square matrix).
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_lines)]
fn gesdd_tall<A, SA, SU, SVT>(
    job: Job,
    a: &mut ArrayBase<SA, Ix2>,
    s: &mut [A::Real],
    u: &mut ArrayBase<SU, Ix2>,
    vt: &mut ArrayBase<SVT, Ix2>,
    work: &mut [A::Real],
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
    let minmn = n; // Since m >= n

    let wantu = job != Job::None;
    let wantvt = job != Job::None;

    // Reduce A to bidiagonal form
    let (d, e, tau_q, tau_p) = gebrd_tall(a);

    // Copy bidiagonal elements
    for (i, val) in d.iter().enumerate() {
        s[i] = *val;
    }
    let mut e_work: Vec<A::Real> = e.to_vec();

    // Generate U if wanted
    if wantu {
        match job {
            Job::All => {
                // U is M x M
                lacpy::lower(a, u);
                q_tall(u, &tau_q);
            }
            Job::Some => {
                // U is M x min(M,N) = M x N
                lacpy::lower(&a.slice(s![.., ..minmn]), &mut u.slice_mut(s![.., ..minmn]));
                let mut u_slice = u.slice_mut(s![.., ..minmn]);
                q_tall(&mut u_slice, &tau_q);
            }
            Job::Overwrite => {
                // U is overwritten in A (M x N)
                // We need to work with a copy since we still need the bidiagonal info
                let mut u_copy: Array2<A> = a.slice(s![.., ..minmn]).to_owned();
                lacpy::lower(&a.slice(s![.., ..minmn]), &mut u_copy);
                q_tall(&mut u_copy, &tau_q);
                // Copy back to A
                for i in 0..m {
                    for j in 0..n {
                        a[(i, j)] = u_copy[(i, j)];
                    }
                }
            }
            Job::None => {}
        }
    }

    // Generate VT if wanted
    if wantvt && job != Job::Overwrite {
        // Copy the upper triangular part of A to VT
        lacpy::upper(&a.slice(s![..n, ..n]), vt);
        p_square(vt, &tau_p);
    } else if wantvt {
        // For Overwrite case, VT is always computed to vt (since U is in A)
        lacpy::upper(&a.slice(s![..n, ..n]), vt);
        p_square(vt, &tau_p);
    }

    // Perform bidiagonal SVD
    let mut c_dummy: Array2<A> = Array2::zeros((minmn, 0));

    match job {
        Job::All | Job::Some => {
            bdsqr::<A, _, _, _>(Uplo::Upper, s, &mut e_work, vt, u, &mut c_dummy, work)
        }
        Job::Overwrite => {
            // U is stored in A
            bdsqr::<A, _, _, _>(Uplo::Upper, s, &mut e_work, vt, a, &mut c_dummy, work)
        }
        Job::None => {
            let mut vt_dummy: Array2<A> = Array2::zeros((minmn, 0));
            let mut u_dummy: Array2<A> = Array2::zeros((0, minmn));
            bdsqr::<A, _, _, _>(
                Uplo::Upper,
                s,
                &mut e_work,
                &mut vt_dummy,
                &mut u_dummy,
                &mut c_dummy,
                work,
            )
        }
    }
}

/// Handle the case when m < n (wide matrix).
#[allow(clippy::many_single_char_names)]
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_lines)]
fn gesdd_wide<A, SA, SU, SVT>(
    job: Job,
    a: &mut ArrayBase<SA, Ix2>,
    s: &mut [A::Real],
    u: &mut ArrayBase<SU, Ix2>,
    vt: &mut ArrayBase<SVT, Ix2>,
    work: &mut [A::Real],
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

    let wantu = job != Job::None;
    let wantvt = job != Job::None;

    // First reduce to bidiagonal form via LQ factorization
    // Compute LQ factorization of A
    let tau_lq = gelqf(a);

    // Copy L to U if needed (L is the lower triangular part of A)
    if wantu && job != Job::Overwrite {
        for i in 0..m {
            for j in 0..=cmp::min(i, m - 1) {
                u[(i, j)] = a[(i, j)];
            }
            for j in (cmp::min(i, m - 1) + 1)..u.ncols() {
                u[(i, j)] = A::zero();
            }
        }
    }

    // Generate Q from LQ factorization into VT if needed
    if wantvt {
        match job {
            Job::All => {
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
            Job::Some => {
                // VT is min(M,N) x N = M x N
                for i in 0..m {
                    for j in 0..n {
                        vt[(i, j)] = a[(i, j)];
                    }
                }
                unglq(vt, &tau_lq);
            }
            Job::Overwrite => {
                // VT is overwritten in A (M x N)
                // But first we need to save L for bidiagonalization
                // Copy L to a temporary matrix
                let mut l_matrix: Array2<A> = Array2::zeros((m, m));
                for i in 0..m {
                    for j in 0..=i {
                        l_matrix[(i, j)] = a[(i, j)];
                    }
                }

                // Generate Q in place of A
                unglq(a, &tau_lq);

                // Now bidiagonalize L and update
                let (d, e, tau_q, _tau_p) = gebrd_tall(&mut l_matrix);

                // Copy bidiagonal elements
                for (i, val) in d.iter().enumerate() {
                    s[i] = *val;
                }
                let mut e_work: Vec<A::Real> = e.to_vec();

                // Update U if needed
                q_tall(&mut l_matrix, &tau_q);
                // Copy to U
                for i in 0..m {
                    for j in 0..m {
                        u[(i, j)] = l_matrix[(i, j)];
                    }
                }

                // Perform bidiagonal SVD with VT = A (which now contains Q)
                let mut c_dummy: Array2<A> = Array2::zeros((m, 0));
                let mut a_slice = a.slice_mut(s![..m, ..]);
                return bdsqr::<A, _, _, _>(
                    Uplo::Upper,
                    s,
                    &mut e_work,
                    &mut a_slice,
                    u,
                    &mut c_dummy,
                    work,
                );
            }
            Job::None => {}
        }
    }

    // Now reduce L (which is m x m) to bidiagonal form
    // L is stored in the lower triangular part of A (and copied to U if needed)
    // We need to work with L separately

    // Extract L into a separate matrix for bidiagonalization
    let mut l_matrix: Array2<A> = Array2::zeros((m, m));
    for i in 0..m {
        for j in 0..=i {
            l_matrix[(i, j)] = if wantu && job != Job::Overwrite {
                u[(i, j)]
            } else {
                a[(i, j)]
            };
        }
    }

    // Reduce L to bidiagonal form
    let (d, e, tau_q, _tau_p) = gebrd_tall(&mut l_matrix);

    // Copy bidiagonal elements
    for (i, val) in d.iter().enumerate() {
        s[i] = *val;
    }
    let mut e_work: Vec<A::Real> = e.to_vec();

    // Update U if needed: U = U * Q_bidiag
    if wantu && job != Job::Overwrite {
        q_tall(&mut l_matrix, &tau_q);
        // Copy back to U
        for i in 0..m {
            for j in 0..m {
                u[(i, j)] = l_matrix[(i, j)];
            }
        }
    }

    // Perform bidiagonal SVD
    let mut c_dummy: Array2<A> = Array2::zeros((m, 0));

    let result = match job {
        Job::All | Job::Some => {
            let mut vt_small = vt.slice_mut(s![..m, ..]);
            bdsqr::<A, _, _, _>(
                Uplo::Upper,
                s,
                &mut e_work,
                &mut vt_small,
                u,
                &mut c_dummy,
                work,
            )
        }
        Job::Overwrite => {
            // Already handled above, but this shouldn't be reached
            unreachable!("Overwrite case should be handled above")
        }
        Job::None => {
            let mut vt_dummy: Array2<A> = Array2::zeros((m, 0));
            let mut u_dummy: Array2<A> = Array2::zeros((0, m));
            bdsqr::<A, _, _, _>(
                Uplo::Upper,
                s,
                &mut e_work,
                &mut vt_dummy,
                &mut u_dummy,
                &mut c_dummy,
                work,
            )
        }
    };

    result
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

        let result = gesdd(Job::All, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_single_element() {
        let mut a = arr2(&[[3.0f64]]);
        let mut s = vec![0.0f64];
        let mut u = Array2::<f64>::zeros((1, 1));
        let mut vt = Array2::<f64>::zeros((1, 1));

        let result = gesdd(Job::All, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());
        assert_abs_diff_eq!(s[0], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_identity_2x2() {
        let mut a = arr2(&[[1.0f64, 0.0], [0.0, 1.0]]);
        let mut s = vec![0.0f64; 2];
        let mut u = Array2::<f64>::zeros((2, 2));
        let mut vt = Array2::<f64>::zeros((2, 2));

        let result = gesdd(Job::All, &mut a, &mut s, &mut u, &mut vt);
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

        let result = gesdd(Job::All, &mut a, &mut s, &mut u, &mut vt);
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

        let result = gesdd(Job::All, &mut a, &mut s, &mut u, &mut vt);
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

        let result = gesdd(Job::All, &mut a, &mut s, &mut u, &mut vt);
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

        let result = gesdd(Job::All, &mut a, &mut s, &mut u, &mut vt);
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
    fn test_job_none() {
        let mut a = arr2(&[[1.0f64, 2.0], [3.0, 4.0]]);
        let mut s = vec![0.0f64; 2];
        let mut u = Array2::<f64>::zeros((0, 0));
        let mut vt = Array2::<f64>::zeros((0, 0));

        let result = gesdd(Job::None, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());

        // Check singular values are computed correctly
        assert!(s[0] > 0.0);
        assert!(s[1] > 0.0);
        assert!(s[0] >= s[1]);
    }

    #[test]
    fn test_job_some_square() {
        let mut a = arr2(&[[1.0f64, 2.0], [3.0, 4.0]]);
        let a_orig = a.clone();
        let mut s = vec![0.0f64; 2];
        let mut u = Array2::<f64>::zeros((2, 2));
        let mut vt = Array2::<f64>::zeros((2, 2));

        let result = gesdd(Job::Some, &mut a, &mut s, &mut u, &mut vt);
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
    fn test_job_some_tall() {
        // 4x2 matrix with Job::Some
        let mut a = arr2(&[[1.0f64, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
        let a_orig = a.clone();
        let mut s = vec![0.0f64; 2];
        let mut u = Array2::<f64>::zeros((4, 2)); // M x min(M,N)
        let mut vt = Array2::<f64>::zeros((2, 2)); // min(M,N) x N

        let result = gesdd(Job::Some, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());

        // Reconstruct A = U * diag(s) * VT
        let mut s_diag = Array2::<f64>::zeros((2, 2));
        s_diag[(0, 0)] = s[0];
        s_diag[(1, 1)] = s[1];
        let reconstructed = u.dot(&s_diag).dot(&vt);

        for i in 0..4 {
            for j in 0..2 {
                assert_abs_diff_eq!(reconstructed[(i, j)], a_orig[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_job_some_wide() {
        // 2x4 matrix with Job::Some
        let mut a = arr2(&[[1.0f64, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
        let a_orig = a.clone();
        let mut s = vec![0.0f64; 2];
        let mut u = Array2::<f64>::zeros((2, 2)); // M x min(M,N)
        let mut vt = Array2::<f64>::zeros((2, 4)); // min(M,N) x N

        let result = gesdd(Job::Some, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());

        // Reconstruct A = U * diag(s) * VT
        let mut s_diag = Array2::<f64>::zeros((2, 2));
        s_diag[(0, 0)] = s[0];
        s_diag[(1, 1)] = s[1];
        let reconstructed = u.dot(&s_diag).dot(&vt);

        for i in 0..2 {
            for j in 0..4 {
                assert_abs_diff_eq!(reconstructed[(i, j)], a_orig[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_orthogonality_u() {
        let mut a = arr2(&[[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]);
        let mut s = vec![0.0f64; 3];
        let mut u = Array2::<f64>::zeros((3, 3));
        let mut vt = Array2::<f64>::zeros((3, 3));

        let result = gesdd(Job::All, &mut a, &mut s, &mut u, &mut vt);
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

        let result = gesdd(Job::All, &mut a, &mut s, &mut u, &mut vt);
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

        let result = gesdd(Job::All, &mut a, &mut s, &mut u, &mut vt);
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

        let result = gesdd(Job::All, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());

        // One singular value should be near zero (rank deficient)
        assert!(s[0] > 0.0);
        assert!(s[1].abs() < 1e-10);
    }

    #[test]
    fn test_3x3_matrix() {
        let mut a = arr2(&[[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let a_orig = a.clone();
        let mut s = vec![0.0f64; 3];
        let mut u = Array2::<f64>::zeros((3, 3));
        let mut vt = Array2::<f64>::zeros((3, 3));

        let result = gesdd(Job::All, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());

        // This matrix has rank 2, so one singular value should be very small
        assert!(s[0] > 0.0);
        assert!(s[1] > 0.0);
        assert!(s[2].abs() < 1e-10);

        // Reconstruct A = U * diag(s) * VT
        let mut s_diag = Array2::<f64>::zeros((3, 3));
        s_diag[(0, 0)] = s[0];
        s_diag[(1, 1)] = s[1];
        s_diag[(2, 2)] = s[2];
        let reconstructed = u.dot(&s_diag).dot(&vt);

        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(reconstructed[(i, j)], a_orig[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_larger_matrix() {
        // 5x4 matrix
        let mut a = arr2(&[
            [1.0f64, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
            [17.0, 18.0, 19.0, 20.0],
        ]);
        let a_orig = a.clone();
        let mut s = vec![0.0f64; 4];
        let mut u = Array2::<f64>::zeros((5, 5));
        let mut vt = Array2::<f64>::zeros((4, 4));

        let result = gesdd(Job::All, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());

        // Check singular values are sorted
        for i in 1..4 {
            assert!(s[i - 1] >= s[i]);
        }

        // Reconstruct A = U[:, :4] * diag(s) * VT
        let mut s_diag = Array2::<f64>::zeros((4, 4));
        for i in 0..4 {
            s_diag[(i, i)] = s[i];
        }
        let u_reduced = u.slice(s![.., ..4]);
        let reconstructed = u_reduced.dot(&s_diag).dot(&vt);

        for i in 0..5 {
            for j in 0..4 {
                assert_abs_diff_eq!(reconstructed[(i, j)], a_orig[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_f32_support() {
        let mut a = arr2(&[[1.0f32, 2.0], [3.0, 4.0]]);
        let a_orig = a.clone();
        let mut s = vec![0.0f32; 2];
        let mut u = Array2::<f32>::zeros((2, 2));
        let mut vt = Array2::<f32>::zeros((2, 2));

        let result = gesdd(Job::All, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());

        // Reconstruct A = U * diag(s) * VT
        let mut s_diag = Array2::<f32>::zeros((2, 2));
        s_diag[(0, 0)] = s[0];
        s_diag[(1, 1)] = s[1];
        let reconstructed = u.dot(&s_diag).dot(&vt);

        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(reconstructed[(i, j)], a_orig[(i, j)], epsilon = 1e-5);
            }
        }
    }
}
