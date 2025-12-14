use std::cmp;
use std::ops::{AddAssign, Div, Mul, MulAssign, Sub};

use ndarray::{s, Array2, ArrayBase, DataMut, Ix2};
use num_traits::{Float, One, Zero};

use super::bdsqr::{bdsqr, Uplo};
use super::gebrd::tall as gebrd_tall;
use super::gelqf::gelqf;
use super::geqrf::geqrf;
use super::lacpy;
use super::lange::maxabs;
use super::lascl;
use super::ungbr::{p_square, q_tall};
use super::unglq::unglq;
use super::ungqr::ungqr;
use crate::{Real, Scalar};

/// Threshold multiplier for deciding when to use QR/LQ factorization.
/// When M >= `MNTHR_MULT` * N (for tall) or N >= `MNTHR_MULT` * M (for wide),
/// use the factorization-based path.
const MNTHR_MULT: usize = 2;

/// Specifies how to compute left singular vectors (U).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

// Keep all enum variants "used" so clippy does not warn when the SVD API is enabled
// but only a subset of options are exercised by the public wrappers.
const _: () = {
    let _ = JobU::Some;
    let _ = JobU::Overwrite;
    let _ = JobVT::Some;
    let _ = JobVT::Overwrite;
};

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

    // Compute threshold for path selection
    // Following LAPACK: use QR/LQ when dimensions are sufficiently different
    let mnthr = MNTHR_MULT * minmn;

    if m >= n {
        // A has at least as many rows as columns
        if m >= mnthr {
            // Path for very tall matrices: use QR factorization first
            // A = Q * R, then bidiagonalize R (which is N x N)
            gesvd_qr_path(jobu, jobvt, a, s, u, vt, &mut work)?;
        } else {
            // Path for nearly-square tall matrices: direct bidiagonalization
            gesvd_tall_direct(jobu, jobvt, a, s, u, vt, &mut work)?;
        }
    } else {
        // m < n: More columns than rows
        if n >= mnthr {
            // Path for very wide matrices: use LQ factorization first
            // A = L * Q, then bidiagonalize L (which is M x M)
            gesvd_lq_path(jobu, jobvt, a, s, u, vt, &mut work)?;
        } else {
            // Path for nearly-square wide matrices: direct bidiagonalization
            // Transpose and use tall path, then swap U and VT
            gesvd_wide_direct(jobu, jobvt, a, s, u, vt, &mut work)?;
        }
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

/// QR path for very tall matrices (M >= MNTHR * N).
/// Computes A = Q * R via QR factorization, then bidiagonalizes R.
#[allow(clippy::many_single_char_names, clippy::similar_names)]
fn gesvd_qr_path<A, SA, SU, SVT>(
    jobu: JobU,
    jobvt: JobVT,
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

    let wantu = jobu != JobU::None;
    let wantvt = jobvt != JobVT::None;

    // Step 1: Compute QR factorization A = Q * R
    let tau_qr = geqrf(a);

    // Step 2: Copy R (upper triangular part) to a separate matrix for bidiagonalization
    let mut r_matrix: Array2<A> = Array2::zeros((n, n));
    for i in 0..n {
        for j in i..n {
            r_matrix[(i, j)] = a[(i, j)];
        }
    }

    // Step 3: If U is wanted, generate Q from the QR factorization
    if wantu {
        match jobu {
            JobU::All => {
                // U is M x M, need full Q
                // Copy the Householder vectors from A to U
                for i in 0..m {
                    for j in 0..n {
                        u[(i, j)] = a[(i, j)];
                    }
                    for j in n..m {
                        u[(i, j)] = A::zero();
                    }
                }
                // Generate full M x M orthogonal matrix
                // ungqr will handle extending to a full orthogonal matrix
                ungqr(u, &tau_qr);
            }
            JobU::Some | JobU::Overwrite => {
                // U is M x N, need first N columns of Q
                for i in 0..m {
                    for j in 0..n {
                        u[(i, j)] = a[(i, j)];
                    }
                }
                ungqr(&mut u.slice_mut(s![.., ..n]), &tau_qr);
            }
            JobU::None => {}
        }
    }

    // Step 4: Bidiagonalize R (which is N x N upper triangular)
    let (d, e, tau_q, tau_p) = gebrd_tall(&mut r_matrix);

    // Copy bidiagonal elements to s
    for (i, val) in d.iter().enumerate() {
        s[i] = *val;
    }
    let mut e_work: Vec<A::Real> = e.to_vec();

    // Step 5: Generate VT from P^H of bidiagonalization
    if wantvt {
        lacpy::upper(&r_matrix, vt);
        p_square(vt, &tau_p);
    }

    // Step 6: Apply Q from bidiagonalization to U
    // U = U * Q_bidiag
    if wantu {
        // Generate Q from bidiagonalization in r_matrix
        q_tall(&mut r_matrix, &tau_q);

        // Multiply: U_new = U_qr * Q_bidiag
        // U is M x N (or M x M), Q_bidiag is N x N
        let u_cols = match jobu {
            JobU::All => m,
            _ => n,
        };
        let u_qr = u.slice(s![.., ..n]).to_owned();
        let q_bidiag = r_matrix.slice(s![..n, ..n]);
        let result = u_qr.dot(&q_bidiag);
        for i in 0..m {
            for j in 0..cmp::min(n, u_cols) {
                u[(i, j)] = result[(i, j)];
            }
        }
    }

    // Step 7: Perform bidiagonal SVD
    let mut c_dummy: Array2<A> = Array2::zeros((n, 0));

    if wantu && wantvt {
        bdsqr::<A, _, _, _>(Uplo::Upper, s, &mut e_work, vt, u, &mut c_dummy, work)
    } else if wantu {
        let mut vt_dummy: Array2<A> = Array2::zeros((n, 0));
        bdsqr::<A, _, _, _>(
            Uplo::Upper,
            s,
            &mut e_work,
            &mut vt_dummy,
            u,
            &mut c_dummy,
            work,
        )
    } else if wantvt {
        let mut u_dummy: Array2<A> = Array2::zeros((0, n));
        bdsqr::<A, _, _, _>(
            Uplo::Upper,
            s,
            &mut e_work,
            vt,
            &mut u_dummy,
            &mut c_dummy,
            work,
        )
    } else {
        let mut vt_dummy: Array2<A> = Array2::zeros((n, 0));
        let mut u_dummy: Array2<A> = Array2::zeros((0, n));
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

/// Direct bidiagonalization path for nearly-square tall matrices (M >= N, M < MNTHR * N).
#[allow(clippy::many_single_char_names, clippy::similar_names)]
fn gesvd_tall_direct<A, SA, SU, SVT>(
    jobu: JobU,
    jobvt: JobVT,
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
    let n = a.ncols();
    let minmn = n; // Since m >= n

    let wantu = jobu != JobU::None;
    let wantvt = jobvt != JobVT::None;

    // Reduce A to bidiagonal form directly
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
    let mut c_dummy: Array2<A> = Array2::zeros((minmn, 0));

    if wantu && wantvt {
        bdsqr::<A, _, _, _>(Uplo::Upper, s, &mut e_work, vt, u, &mut c_dummy, work)
    } else if wantu {
        let mut vt_dummy: Array2<A> = Array2::zeros((minmn, 0));
        bdsqr::<A, _, _, _>(
            Uplo::Upper,
            s,
            &mut e_work,
            &mut vt_dummy,
            u,
            &mut c_dummy,
            work,
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
            work,
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
            work,
        )
    }
}

/// LQ path for very wide matrices (N >= MNTHR * M).
/// Computes A = L * Q via LQ factorization, then bidiagonalizes L.
#[allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]
fn gesvd_lq_path<A, SA, SU, SVT>(
    jobu: JobU,
    jobvt: JobVT,
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

    let wantu = jobu != JobU::None;
    let wantvt = jobvt != JobVT::None;

    // Step 1: Compute LQ factorization A = L * Q
    let tau_lq = gelqf(a);

    // Step 2: Copy L (lower triangular part) to a separate matrix for bidiagonalization
    let mut l_matrix: Array2<A> = Array2::zeros((m, m));
    for i in 0..m {
        for j in 0..=i {
            l_matrix[(i, j)] = a[(i, j)];
        }
    }

    // Step 3: If VT is wanted, generate Q from the LQ factorization
    if wantvt {
        match jobvt {
            JobVT::All => {
                // VT is N x N, need full Q
                // Copy the Householder vectors from A to VT
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
                // Generate full N x N orthogonal matrix
                // unglq will handle extending to a full orthogonal matrix
                unglq(vt, &tau_lq);
            }
            JobVT::Some => {
                // VT is M x N, need first M rows of Q
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

    // Step 4: Bidiagonalize L (which is M x M lower triangular)
    let (d, e, tau_q, tau_p) = gebrd_tall(&mut l_matrix);

    // Copy bidiagonal elements to s
    for (i, val) in d.iter().enumerate() {
        s[i] = *val;
    }
    let mut e_work: Vec<A::Real> = e.to_vec();

    // Step 5: Generate U from Q of bidiagonalization
    if wantu {
        match jobu {
            JobU::All | JobU::Some => {
                lacpy::lower(&l_matrix, u);
                q_tall(u, &tau_q);
            }
            JobU::Overwrite | JobU::None => {}
        }
    }

    // Step 6: Apply P^H from bidiagonalization to VT
    // VT = P^H_bidiag * VT_lq
    if wantvt {
        // Generate P^H from bidiagonalization
        let mut p_matrix: Array2<A> = Array2::zeros((m, m));
        lacpy::upper(&l_matrix, &mut p_matrix);
        p_square(&mut p_matrix, &tau_p);

        // Multiply: VT_new = P^H * VT_lq
        let vt_rows = match jobvt {
            JobVT::All => n,
            _ => m,
        };
        let vt_lq = vt.slice(s![..m, ..]).to_owned();
        let result = p_matrix.dot(&vt_lq);
        for i in 0..m {
            for j in 0..n {
                vt[(i, j)] = result[(i, j)];
            }
        }
        // Rows m..vt_rows remain unchanged (already set to identity extension)
        let _ = vt_rows;
    }

    // Step 7: Perform bidiagonal SVD
    let mut c_dummy: Array2<A> = Array2::zeros((m, 0));

    if wantu && wantvt {
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
    } else if wantu {
        let mut vt_dummy: Array2<A> = Array2::zeros((m, 0));
        bdsqr::<A, _, _, _>(
            Uplo::Upper,
            s,
            &mut e_work,
            &mut vt_dummy,
            u,
            &mut c_dummy,
            work,
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
            work,
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
            work,
        )
    }
}

/// Direct bidiagonalization path for nearly-square wide matrices (M < N, N < MNTHR * M).
/// Uses LQ factorization followed by bidiagonalization of L.
#[allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]
fn gesvd_wide_direct<A, SA, SU, SVT>(
    jobu: JobU,
    jobvt: JobVT,
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

    let wantu = jobu != JobU::None;
    let wantvt = jobvt != JobVT::None;

    // For nearly-square wide matrices, we still use LQ factorization
    // but the cost savings vs direct bidiagonalization are smaller.
    // The algorithm is the same as gesvd_lq_path.

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
                // Generate full N x N orthogonal matrix
                unglq(vt, &tau_lq);
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

    // Extract L into a separate matrix for bidiagonalization
    let mut l_matrix: Array2<A> = Array2::zeros((m, m));
    for i in 0..m {
        for j in 0..=i {
            l_matrix[(i, j)] = if wantu { u[(i, j)] } else { a[(i, j)] };
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
    if wantu {
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

    if wantu && wantvt {
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
    } else if wantu {
        let mut vt_dummy: Array2<A> = Array2::zeros((m, 0));
        bdsqr::<A, _, _, _>(
            Uplo::Upper,
            s,
            &mut e_work,
            &mut vt_dummy,
            u,
            &mut c_dummy,
            work,
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
            work,
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
            work,
        )
    }
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
    fn test_jobu_some_jobvt_some() {
        let mut a = arr2(&[[1.0f64, 0.0], [0.0, 2.0]]);
        let mut s = vec![0.0f64; 2];
        let mut u = Array2::<f64>::zeros((2, 2));
        let mut vt = Array2::<f64>::zeros((2, 2));

        let result = gesvd(JobU::Some, JobVT::Some, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());
        assert_abs_diff_eq!(s[0], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_jobu_overwrite_jobvt_overwrite() {
        let mut a = arr2(&[[3.0f64, 0.0], [0.0, 1.0]]);
        let mut s = vec![0.0f64; 2];
        let mut u = Array2::<f64>::zeros((2, 2));
        let mut vt = Array2::<f64>::zeros((2, 2));

        let result = gesvd(
            JobU::Overwrite,
            JobVT::Overwrite,
            &mut a,
            &mut s,
            &mut u,
            &mut vt,
        );
        assert!(result.is_ok());
        assert_abs_diff_eq!(s[0], 3.0, epsilon = 1e-10);
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

    #[test]
    fn test_very_tall_matrix_qr_path() {
        // 6x2 matrix triggers QR path (M >= MNTHR * N = 2 * 2 = 4)
        let mut a = arr2(&[
            [1.0f64, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ]);
        let a_orig = a.clone();
        let mut s = vec![0.0f64; 2];
        let mut u = Array2::<f64>::zeros((6, 6));
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

        for i in 0..6 {
            for j in 0..2 {
                assert_abs_diff_eq!(reconstructed[(i, j)], a_orig[(i, j)], epsilon = 1e-8);
            }
        }

        // Check orthogonality of U
        let utu = u.t().dot(&u);
        for i in 0..6 {
            for j in 0..6 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(utu[(i, j)], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_very_wide_matrix_lq_path() {
        // 2x6 matrix triggers LQ path (N >= MNTHR * M = 2 * 2 = 4)
        let mut a = arr2(&[
            [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ]);
        let a_orig = a.clone();
        let mut s = vec![0.0f64; 2];
        let mut u = Array2::<f64>::zeros((2, 2));
        let mut vt = Array2::<f64>::zeros((6, 6));

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
            for j in 0..6 {
                assert_abs_diff_eq!(reconstructed[(i, j)], a_orig[(i, j)], epsilon = 1e-8);
            }
        }

        // Check orthogonality of VT
        let vtvt = vt.dot(&vt.t());
        for i in 0..6 {
            for j in 0..6 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(vtvt[(i, j)], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_tall_direct_path() {
        // 3x2 matrix uses direct path (M < MNTHR * N = 2 * 2 = 4)
        let mut a = arr2(&[[1.0f64, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let a_orig = a.clone();
        let mut s = vec![0.0f64; 2];
        let mut u = Array2::<f64>::zeros((3, 3));
        let mut vt = Array2::<f64>::zeros((2, 2));

        let result = gesvd(JobU::All, JobVT::All, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());

        // Reconstruct A = U[:, :2] * diag(s) * VT
        let mut s_diag = Array2::<f64>::zeros((2, 2));
        s_diag[(0, 0)] = s[0];
        s_diag[(1, 1)] = s[1];
        let u_reduced = u.slice(s![.., ..2]);
        let reconstructed = u_reduced.dot(&s_diag).dot(&vt);

        for i in 0..3 {
            for j in 0..2 {
                assert_abs_diff_eq!(reconstructed[(i, j)], a_orig[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_wide_direct_path() {
        // 2x3 matrix uses direct path (N < MNTHR * M = 2 * 2 = 4)
        let mut a = arr2(&[[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let a_orig = a.clone();
        let mut s = vec![0.0f64; 2];
        let mut u = Array2::<f64>::zeros((2, 2));
        let mut vt = Array2::<f64>::zeros((3, 3));

        let result = gesvd(JobU::All, JobVT::All, &mut a, &mut s, &mut u, &mut vt);
        assert!(result.is_ok());

        // Reconstruct A = U * diag(s) * VT[:2, :]
        let mut s_diag = Array2::<f64>::zeros((2, 2));
        s_diag[(0, 0)] = s[0];
        s_diag[(1, 1)] = s[1];
        let vt_reduced = vt.slice(s![..2, ..]);
        let reconstructed = u.dot(&s_diag).dot(&vt_reduced);

        for i in 0..2 {
            for j in 0..3 {
                assert_abs_diff_eq!(reconstructed[(i, j)], a_orig[(i, j)], epsilon = 1e-8);
            }
        }
    }
}
