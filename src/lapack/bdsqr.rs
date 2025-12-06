use std::ops::{AddAssign, Mul, Sub};

use ndarray::{ArrayBase, DataMut, Ix2};
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};

use super::lartg::lartg;
use super::las2::las2;
use super::lasq1::lasq1;
use super::lasr::{lasr, Direction, Pivot, Side};
use super::lasv2::lasv2;
use crate::blas::scal::rscal;
use crate::Real;
use crate::Scalar;

/// Specifies whether the bidiagonal matrix is upper or lower bidiagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Uplo {
    /// Upper bidiagonal: has nonzero elements on diagonal and superdiagonal
    Upper,
    /// Lower bidiagonal: has nonzero elements on diagonal and subdiagonal
    Lower,
}

/// Computes the singular value decomposition of a real bidiagonal matrix.
///
/// `bdsqr` computes the singular values and, optionally, the right and/or
/// left singular vectors from the singular value decomposition (SVD) of
/// a real N-by-N (upper or lower) bidiagonal matrix B using the implicit
/// zero-shift QR algorithm.
///
/// The SVD of B has the form:
///    B = Q * S * P^H
/// where S is the diagonal matrix of singular values, Q is an orthogonal
/// matrix of left singular vectors, and P is an orthogonal matrix of
/// right singular vectors. If left singular vectors are requested, this
/// subroutine actually returns U*Q instead of Q, and, if right singular
/// vectors are requested, this subroutine returns P^H*VT instead of P^H,
/// for given complex input matrices U and VT.
///
/// # Arguments
///
/// * `uplo` - Specifies whether B is upper or lower bidiagonal.
/// * `d` - On entry, the N diagonal elements of the bidiagonal matrix B.
///   On exit, if Ok, the singular values of B in decreasing order.
/// * `e` - On entry, the N-1 off-diagonal elements of the bidiagonal matrix B.
///   On exit, the contents are overwritten.
/// * `vt` - On entry, an NCVT-by-N matrix VT. On exit, VT is overwritten by P^H * VT.
///   Not referenced if NCVT = 0 (i.e., if `vt` has 0 columns).
/// * `u` - On entry, an N-by-NRU matrix U. On exit, U is overwritten by U * Q.
///   Not referenced if NRU = 0 (i.e., if `u` has 0 rows).
/// * `c` - On entry, an N-by-NCC matrix C. On exit, C is overwritten by Q^H * C.
///   Not referenced if NCC = 0 (i.e., if `c` has 0 columns).
/// * `work` - Workspace array of length at least 4*N.
///
/// # Returns
///
/// * `Ok(())` if successful.
/// * `Err(info)` where `info > 0` means the algorithm did not converge;
///   `info` specifies how many superdiagonals of an intermediate bidiagonal
///   form B did not converge to zero.
#[allow(dead_code)]
#[allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
pub fn bdsqr<A, SVT, SU, SC>(
    uplo: Uplo,
    d: &mut [A::Real],
    e: &mut [A::Real],
    vt: &mut ArrayBase<SVT, Ix2>,
    u: &mut ArrayBase<SU, Ix2>,
    c: &mut ArrayBase<SC, Ix2>,
    work: &mut [A::Real],
) -> Result<(), usize>
where
    A: Scalar + AddAssign + Mul<A::Real, Output = A> + Sub<Output = A>,
    A::Real: Real,
    SVT: DataMut<Elem = A>,
    SU: DataMut<Elem = A>,
    SC: DataMut<Elem = A>,
{
    let n = d.len();
    let ncvt = vt.dim().1;
    let nru = u.dim().0;
    let ncc = c.dim().1;

    let zero = A::Real::zero();
    let one = A::Real::one();
    let neg_one = -one;

    // Quick return if possible
    if n == 0 {
        return Ok(());
    }

    if n == 1 {
        // Handle single element case
        if d[0] < zero {
            d[0] = -d[0];
            if ncvt > 0 {
                rscal::<A, _>(ncvt, neg_one, &mut vt.row_mut(0));
            }
        }
        return Ok(());
    }

    // ROTATE is true if any singular vectors desired, false otherwise
    let rotate = ncvt > 0 || nru > 0 || ncc > 0;

    // If no singular vectors desired, use qd algorithm
    if !rotate {
        let result = lasq1(d, e, work);

        // If INFO equals 2, dqds didn't finish, try to finish
        if result != Err(2) {
            return result.map_err(|err| err as usize);
        }
    }

    let nm1 = n - 1;
    let nm12 = nm1 + nm1;
    let nm13 = nm12 + nm1;

    // Get machine constants
    let eps = A::Real::eps();
    let unfl = A::Real::sfmin();

    let lower = uplo == Uplo::Lower;

    // If matrix is lower bidiagonal, rotate to be upper bidiagonal
    // by applying Givens rotations on the left
    if lower {
        for i in 0..n - 1 {
            let (cs, sn, r) = lartg(d[i], e[i]);
            d[i] = r;
            e[i] = sn * d[i + 1];
            d[i + 1] = cs * d[i + 1];
            work[i] = cs;
            work[nm1 + i] = sn;
        }

        // Update singular vectors if desired
        if nru > 0 {
            lasr::<A, _>(
                Side::Right,
                Pivot::Variable,
                Direction::Forward,
                &work[..nm1],
                &work[nm1..nm1 + nm1],
                u,
            );
        }
        if ncc > 0 {
            lasr::<A, _>(
                Side::Left,
                Pivot::Variable,
                Direction::Forward,
                &work[..nm1],
                &work[nm1..nm1 + nm1],
                c,
            );
        }
    }

    // Compute singular values to relative accuracy TOL
    // (By setting TOL to be negative, algorithm will compute
    // singular values to absolute accuracy ABS(TOL)*norm(input matrix))
    let ten: A::Real = FromPrimitive::from_f64(10.0).unwrap();
    let hndrd: A::Real = FromPrimitive::from_f64(100.0).unwrap();
    let meigth: A::Real = FromPrimitive::from_f64(-0.125).unwrap();
    let hndrth: A::Real = FromPrimitive::from_f64(0.01).unwrap();
    let maxitr: A::Real = FromPrimitive::from_f64(6.0).unwrap();

    let tolmul = ten.max(hndrd.min(eps.powf(meigth)));
    let tol = tolmul * eps;

    // Compute approximate maximum, minimum singular values
    let mut smax = zero;
    for val in d.iter().take(n) {
        smax = smax.max(val.abs());
    }
    for val in e.iter().take(n - 1) {
        smax = smax.max(val.abs());
    }

    let thresh;

    let n_real: A::Real = FromPrimitive::from_usize(n).unwrap();

    if tol >= zero {
        // Relative accuracy desired
        let mut sminoa = d[0].abs();
        if sminoa != zero {
            let mut mu = sminoa;
            for i in 1..n {
                mu = d[i].abs() * (mu / (mu + e[i - 1].abs()));
                sminoa = sminoa.min(mu);
                if sminoa == zero {
                    break;
                }
            }
            sminoa /= n_real.sqrt();
        }
        thresh = (tol * sminoa).max(maxitr * (n_real * (n_real * unfl)));
    } else {
        // Absolute accuracy desired
        thresh = (tol.abs() * smax).max(maxitr * (n_real * (n_real * unfl)));
    }

    // Prepare for main iteration loop for the singular values
    let maxitdivn = (maxitr * n_real).to_usize().unwrap();
    let mut iterdivn = 0;
    let mut iter = 0_isize;
    let mut oldll = usize::MAX;
    let mut oldm = usize::MAX;

    // M points to last element of unconverged part of matrix
    let mut m = n;

    // Begin main iteration loop
    let mut idir = 0;

    loop {
        // Check for convergence or exceeding iteration count
        if m <= 1 {
            break;
        }

        if iter >= n as isize {
            iter -= n as isize;
            iterdivn += 1;
            if iterdivn >= maxitdivn {
                // Maximum number of iterations exceeded
                let mut info = 0;
                for val in e.iter().take(n - 1) {
                    if *val != zero {
                        info += 1;
                    }
                }
                return Err(info);
            }
        }

        // Find diagonal block of matrix to work on
        if tol < zero && d[m - 1].abs() <= thresh {
            d[m - 1] = zero;
        }
        let mut ll = 0;
        smax = d[m - 1].abs();

        let mut found_split = false;
        for lll in 1..m {
            ll = m - lll - 1;
            let abss = d[ll].abs();
            let abse = e[ll].abs();
            if tol < zero && abss <= thresh {
                d[ll] = zero;
            }
            if abse <= thresh {
                found_split = true;
                break;
            }
            smax = smax.max(abss).max(abse);
        }

        if found_split {
            e[ll] = zero;

            // Matrix splits since E(LL) = 0
            if ll == m - 2 {
                // Convergence of bottom singular value
                m -= 1;
                continue;
            }
            ll += 1;
        } else {
            ll = 0;
        }
        // E(LL) through E(M-1) are nonzero, E(LL-1) is zero (if LL > 0)

        if ll == m - 2 {
            // 2 by 2 block, handle separately
            let (sigmn, sigmx, sinr, cosr, sinl, cosl) = lasv2(d[m - 2], e[m - 2], d[m - 1]);
            d[m - 2] = sigmx;
            e[m - 2] = zero;
            d[m - 1] = sigmn;

            // Compute singular vectors, if desired
            // Apply rotation in-place to avoid double mutable borrow
            if ncvt > 0 {
                for j in 0..ncvt {
                    let temp = vt[(m - 2, j)] * cosr.into() + vt[(m - 1, j)] * sinr.into();
                    vt[(m - 1, j)] = vt[(m - 1, j)] * cosr.into() - vt[(m - 2, j)] * sinr.into();
                    vt[(m - 2, j)] = temp;
                }
            }
            if nru > 0 {
                for i in 0..nru {
                    let temp = u[(i, m - 2)] * cosl.into() + u[(i, m - 1)] * sinl.into();
                    u[(i, m - 1)] = u[(i, m - 1)] * cosl.into() - u[(i, m - 2)] * sinl.into();
                    u[(i, m - 2)] = temp;
                }
            }
            if ncc > 0 {
                for j in 0..ncc {
                    let temp = c[(m - 2, j)] * cosl.into() + c[(m - 1, j)] * sinl.into();
                    c[(m - 1, j)] = c[(m - 1, j)] * cosl.into() - c[(m - 2, j)] * sinl.into();
                    c[(m - 2, j)] = temp;
                }
            }
            m -= 2;
            continue;
        }

        // If working on new submatrix, choose shift direction
        // (from larger end diagonal element towards smaller)
        if ll > oldm || m < oldll {
            if d[ll].abs() >= d[m - 1].abs() {
                // Chase bulge from top (big end) to bottom (small end)
                idir = 1;
            } else {
                // Chase bulge from bottom (big end) to top (small end)
                idir = 2;
            }
        }

        // Apply convergence tests
        let mut smin = zero;
        if idir == 1 {
            // Run convergence test in forward direction
            // First apply standard test to bottom of matrix
            if e[m - 2].abs() <= tol.abs() * d[m - 1].abs()
                || (tol < zero && e[m - 2].abs() <= thresh)
            {
                e[m - 2] = zero;
                continue;
            }

            if tol >= zero {
                // If relative accuracy desired, apply convergence criterion forward
                let mut mu = d[ll].abs();
                smin = mu;
                let mut converged = false;
                for lll in ll..m - 1 {
                    if e[lll].abs() <= tol * mu {
                        e[lll] = zero;
                        converged = true;
                        break;
                    }
                    mu = d[lll + 1].abs() * (mu / (mu + e[lll].abs()));
                    smin = smin.min(mu);
                }
                if converged {
                    continue;
                }
            }
        } else {
            // Run convergence test in backward direction
            // First apply standard test to top of matrix
            if e[ll].abs() <= tol.abs() * d[ll].abs() || (tol < zero && e[ll].abs() <= thresh) {
                e[ll] = zero;
                continue;
            }

            if tol >= zero {
                // If relative accuracy desired, apply convergence criterion backward
                let mut mu = d[m - 1].abs();
                smin = mu;
                let mut converged = false;
                for lll in (ll..m - 1).rev() {
                    if e[lll].abs() <= tol * mu {
                        e[lll] = zero;
                        converged = true;
                        break;
                    }
                    mu = d[lll].abs() * (mu / (mu + e[lll].abs()));
                    smin = smin.min(mu);
                }
                if converged {
                    continue;
                }
            }
        }
        oldll = ll;
        oldm = m;

        // Compute shift. First, test if shifting would ruin relative
        // accuracy, and if so set the shift to zero.
        let shift = if tol >= zero && n_real * tol * (smin / smax) <= eps.max(hndrth * tol) {
            // Use a zero shift to avoid loss of relative accuracy
            zero
        } else {
            // Compute the shift from 2-by-2 block at end of matrix
            let (shift_val, _r) = if idir == 1 {
                las2(d[m - 2], e[m - 2], d[m - 1])
            } else {
                las2(d[ll], e[ll], d[ll + 1])
            };

            // Test if shift negligible, and if so set to zero
            let sll = if idir == 1 {
                d[ll].abs()
            } else {
                d[m - 1].abs()
            };
            if sll > zero && (shift_val / sll).powi(2) < eps {
                zero
            } else {
                shift_val
            }
        };

        // Increment iteration count
        iter += (m - ll) as isize;

        // If SHIFT = 0, do simplified QR iteration
        if shift == zero {
            if idir == 1 {
                // Chase bulge from top to bottom
                // Save cosines and sines for later singular vector updates
                let mut cs = one;
                let mut oldcs = one;
                let mut oldsn = zero;
                for i in ll..m - 1 {
                    let (cs_new, sn, r) = lartg(d[i] * cs, e[i]);
                    cs = cs_new;
                    if i > ll {
                        e[i - 1] = oldsn * r;
                    }
                    let (oldcs_new, oldsn_new, d_new) = lartg(oldcs * r, d[i + 1] * sn);
                    oldcs = oldcs_new;
                    oldsn = oldsn_new;
                    d[i] = d_new;
                    work[i - ll] = cs;
                    work[i - ll + nm1] = sn;
                    work[i - ll + nm12] = oldcs;
                    work[i - ll + nm13] = oldsn;
                }
                let h = d[m - 1] * cs;
                d[m - 1] = h * oldcs;
                e[m - 2] = h * oldsn;

                // Update singular vectors
                if ncvt > 0 {
                    let len = m - ll;
                    lasr::<A, _>(
                        Side::Left,
                        Pivot::Variable,
                        Direction::Forward,
                        &work[..len],
                        &work[nm1..nm1 + len],
                        &mut vt.slice_mut(ndarray::s![ll..m, ..]),
                    );
                }
                if nru > 0 {
                    let len = m - ll;
                    lasr::<A, _>(
                        Side::Right,
                        Pivot::Variable,
                        Direction::Forward,
                        &work[nm12..nm12 + len],
                        &work[nm13..nm13 + len],
                        &mut u.slice_mut(ndarray::s![.., ll..m]),
                    );
                }
                if ncc > 0 {
                    let len = m - ll;
                    lasr::<A, _>(
                        Side::Left,
                        Pivot::Variable,
                        Direction::Forward,
                        &work[nm12..nm12 + len],
                        &work[nm13..nm13 + len],
                        &mut c.slice_mut(ndarray::s![ll..m, ..]),
                    );
                }

                // Test convergence
                if e[m - 2].abs() <= thresh {
                    e[m - 2] = zero;
                }
            } else {
                // Chase bulge from bottom to top
                // Save cosines and sines for later singular vector updates
                let mut cs = one;
                let mut oldcs = one;
                let mut oldsn = zero;
                for i in ((ll + 1)..m).rev() {
                    let (cs_new, sn, r) = lartg(d[i] * cs, e[i - 1]);
                    cs = cs_new;
                    if i < m - 1 {
                        e[i] = oldsn * r;
                    }
                    let (oldcs_new, oldsn_new, d_new) = lartg(oldcs * r, d[i - 1] * sn);
                    oldcs = oldcs_new;
                    oldsn = oldsn_new;
                    d[i] = d_new;
                    work[i - ll - 1] = cs;
                    work[i - ll - 1 + nm1] = -sn;
                    work[i - ll - 1 + nm12] = oldcs;
                    work[i - ll - 1 + nm13] = -oldsn;
                }
                let h = d[ll] * cs;
                d[ll] = h * oldcs;
                e[ll] = h * oldsn;

                // Update singular vectors
                if ncvt > 0 {
                    let len = m - ll;
                    lasr::<A, _>(
                        Side::Left,
                        Pivot::Variable,
                        Direction::Backward,
                        &work[nm12..nm12 + len],
                        &work[nm13..nm13 + len],
                        &mut vt.slice_mut(ndarray::s![ll..m, ..]),
                    );
                }
                if nru > 0 {
                    let len = m - ll;
                    lasr::<A, _>(
                        Side::Right,
                        Pivot::Variable,
                        Direction::Backward,
                        &work[..len],
                        &work[nm1..nm1 + len],
                        &mut u.slice_mut(ndarray::s![.., ll..m]),
                    );
                }
                if ncc > 0 {
                    let len = m - ll;
                    lasr::<A, _>(
                        Side::Left,
                        Pivot::Variable,
                        Direction::Backward,
                        &work[..len],
                        &work[nm1..nm1 + len],
                        &mut c.slice_mut(ndarray::s![ll..m, ..]),
                    );
                }

                // Test convergence
                if e[ll].abs() <= thresh {
                    e[ll] = zero;
                }
            }
        } else {
            // Use nonzero shift
            if idir == 1 {
                // Chase bulge from top to bottom
                // Save cosines and sines for later singular vector updates
                let mut f = (d[ll].abs() - shift) * (sign(one, d[ll]) + shift / d[ll]);
                let mut g = e[ll];
                for i in ll..m - 1 {
                    let (cosr, sinr, r) = lartg(f, g);
                    if i > ll {
                        e[i - 1] = r;
                    }
                    f = cosr * d[i] + sinr * e[i];
                    e[i] = cosr * e[i] - sinr * d[i];
                    g = sinr * d[i + 1];
                    d[i + 1] = cosr * d[i + 1];
                    let (cosl, sinl, r) = lartg(f, g);
                    d[i] = r;
                    f = cosl * e[i] + sinl * d[i + 1];
                    d[i + 1] = cosl * d[i + 1] - sinl * e[i];
                    if i < m - 2 {
                        g = sinl * e[i + 1];
                        e[i + 1] = cosl * e[i + 1];
                    }
                    work[i - ll] = cosr;
                    work[i - ll + nm1] = sinr;
                    work[i - ll + nm12] = cosl;
                    work[i - ll + nm13] = sinl;
                }
                e[m - 2] = f;

                // Update singular vectors
                if ncvt > 0 {
                    let len = m - ll;
                    lasr::<A, _>(
                        Side::Left,
                        Pivot::Variable,
                        Direction::Forward,
                        &work[..len],
                        &work[nm1..nm1 + len],
                        &mut vt.slice_mut(ndarray::s![ll..m, ..]),
                    );
                }
                if nru > 0 {
                    let len = m - ll;
                    lasr::<A, _>(
                        Side::Right,
                        Pivot::Variable,
                        Direction::Forward,
                        &work[nm12..nm12 + len],
                        &work[nm13..nm13 + len],
                        &mut u.slice_mut(ndarray::s![.., ll..m]),
                    );
                }
                if ncc > 0 {
                    let len = m - ll;
                    lasr::<A, _>(
                        Side::Left,
                        Pivot::Variable,
                        Direction::Forward,
                        &work[nm12..nm12 + len],
                        &work[nm13..nm13 + len],
                        &mut c.slice_mut(ndarray::s![ll..m, ..]),
                    );
                }

                // Test convergence
                if e[m - 2].abs() <= thresh {
                    e[m - 2] = zero;
                }
            } else {
                // Chase bulge from bottom to top
                // Save cosines and sines for later singular vector updates
                let mut f = (d[m - 1].abs() - shift) * (sign(one, d[m - 1]) + shift / d[m - 1]);
                let mut g = e[m - 2];
                for i in ((ll + 1)..m).rev() {
                    let (cosr, sinr, r) = lartg(f, g);
                    if i < m - 1 {
                        e[i] = r;
                    }
                    f = cosr * d[i] + sinr * e[i - 1];
                    e[i - 1] = cosr * e[i - 1] - sinr * d[i];
                    g = sinr * d[i - 1];
                    d[i - 1] = cosr * d[i - 1];
                    let (cosl, sinl, r) = lartg(f, g);
                    d[i] = r;
                    f = cosl * e[i - 1] + sinl * d[i - 1];
                    d[i - 1] = cosl * d[i - 1] - sinl * e[i - 1];
                    if i > ll + 1 {
                        g = sinl * e[i - 2];
                        e[i - 2] = cosl * e[i - 2];
                    }
                    work[i - ll - 1] = cosr;
                    work[i - ll - 1 + nm1] = -sinr;
                    work[i - ll - 1 + nm12] = cosl;
                    work[i - ll - 1 + nm13] = -sinl;
                }
                e[ll] = f;

                // Test convergence
                if e[ll].abs() <= thresh {
                    e[ll] = zero;
                }

                // Update singular vectors if desired
                if ncvt > 0 {
                    let len = m - ll;
                    lasr::<A, _>(
                        Side::Left,
                        Pivot::Variable,
                        Direction::Backward,
                        &work[nm12..nm12 + len],
                        &work[nm13..nm13 + len],
                        &mut vt.slice_mut(ndarray::s![ll..m, ..]),
                    );
                }
                if nru > 0 {
                    let len = m - ll;
                    lasr::<A, _>(
                        Side::Right,
                        Pivot::Variable,
                        Direction::Backward,
                        &work[..len],
                        &work[nm1..nm1 + len],
                        &mut u.slice_mut(ndarray::s![.., ll..m]),
                    );
                }
                if ncc > 0 {
                    let len = m - ll;
                    lasr::<A, _>(
                        Side::Left,
                        Pivot::Variable,
                        Direction::Backward,
                        &work[..len],
                        &work[nm1..nm1 + len],
                        &mut c.slice_mut(ndarray::s![ll..m, ..]),
                    );
                }
            }
        }
    }

    // All singular values converged, so make them positive
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        if d[i] == zero {
            // Avoid -ZERO by explicit assignment (no-op in Rust but keeps structure)
        } else if d[i] < zero {
            d[i] = -d[i];

            // Change sign of singular vectors, if desired
            if ncvt > 0 {
                rscal::<A, _>(ncvt, neg_one, &mut vt.row_mut(i));
            }
        }
    }

    // Sort the singular values into decreasing order (insertion sort on
    // singular values, but only one transposition per singular vector)
    #[allow(clippy::needless_range_loop)]
    for i in 0..n - 1 {
        // Scan for smallest D(I)
        let mut isub = 0;
        let mut smin_sort = d[0];
        #[allow(clippy::needless_range_loop)]
        for j in 1..n - i {
            if d[j] <= smin_sort {
                isub = j;
                smin_sort = d[j];
            }
        }
        if isub != n - 1 - i {
            // Swap singular values and vectors
            d[isub] = d[n - 1 - i];
            d[n - 1 - i] = smin_sort;
            // Swap vectors in-place to avoid double mutable borrow
            if ncvt > 0 {
                for j in 0..ncvt {
                    let temp = vt[(isub, j)];
                    vt[(isub, j)] = vt[(n - 1 - i, j)];
                    vt[(n - 1 - i, j)] = temp;
                }
            }
            if nru > 0 {
                for k in 0..nru {
                    let temp = u[(k, isub)];
                    u[(k, isub)] = u[(k, n - 1 - i)];
                    u[(k, n - 1 - i)] = temp;
                }
            }
            if ncc > 0 {
                for j in 0..ncc {
                    let temp = c[(isub, j)];
                    c[(isub, j)] = c[(n - 1 - i, j)];
                    c[(n - 1 - i, j)] = temp;
                }
            }
        }
    }

    Ok(())
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
    use ndarray::{arr2, Array2};
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn test_empty() {
        let mut d: [f64; 0] = [];
        let mut e: [f64; 0] = [];
        let mut vt: Array2<f64> = Array2::zeros((0, 0));
        let mut u: Array2<f64> = Array2::zeros((0, 0));
        let mut c: Array2<f64> = Array2::zeros((0, 0));
        let mut work: [f64; 0] = [];
        let result = bdsqr::<f64, _, _, _>(
            Uplo::Upper,
            &mut d,
            &mut e,
            &mut vt,
            &mut u,
            &mut c,
            &mut work,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_single_element() {
        let mut d = [3.0f64];
        let mut e: [f64; 0] = [];
        let mut vt: Array2<f64> = Array2::zeros((1, 0));
        let mut u: Array2<f64> = Array2::zeros((0, 1));
        let mut c: Array2<f64> = Array2::zeros((1, 0));
        let mut work = [0.0f64; 4];
        let result = bdsqr::<f64, _, _, _>(
            Uplo::Upper,
            &mut d,
            &mut e,
            &mut vt,
            &mut u,
            &mut c,
            &mut work,
        );
        assert!(result.is_ok());
        assert_eq!(d[0], 3.0);
    }

    #[test]
    fn test_single_element_negative() {
        let mut d = [-5.0f64];
        let mut e: [f64; 0] = [];
        let mut vt: Array2<f64> = Array2::zeros((1, 0));
        let mut u: Array2<f64> = Array2::zeros((0, 1));
        let mut c: Array2<f64> = Array2::zeros((1, 0));
        let mut work = [0.0f64; 4];
        let result = bdsqr::<f64, _, _, _>(
            Uplo::Upper,
            &mut d,
            &mut e,
            &mut vt,
            &mut u,
            &mut c,
            &mut work,
        );
        assert!(result.is_ok());
        assert_eq!(d[0], 5.0); // Should take absolute value
    }

    #[test]
    fn test_two_by_two_upper() {
        // Simple 2x2 upper bidiagonal matrix:
        // [3 1]
        // [0 2]
        let mut d = [3.0f64, 2.0];
        let mut e = [1.0f64];
        let mut vt: Array2<f64> = Array2::zeros((2, 0));
        let mut u: Array2<f64> = Array2::zeros((0, 2));
        let mut c: Array2<f64> = Array2::zeros((2, 0));
        let mut work = [0.0f64; 8];
        let result = bdsqr::<f64, _, _, _>(
            Uplo::Upper,
            &mut d,
            &mut e,
            &mut vt,
            &mut u,
            &mut c,
            &mut work,
        );
        assert!(result.is_ok());
        // Singular values should be in descending order
        assert!(d[0] >= d[1]);
        assert!(d[0] > 0.0);
        assert!(d[1] > 0.0);
    }

    #[test]
    fn test_two_by_two_lower() {
        // Simple 2x2 lower bidiagonal matrix:
        // [3 0]
        // [1 2]
        let mut d = [3.0f64, 2.0];
        let mut e = [1.0f64];
        let mut vt: Array2<f64> = Array2::zeros((2, 0));
        let mut u: Array2<f64> = Array2::zeros((0, 2));
        let mut c: Array2<f64> = Array2::zeros((2, 0));
        let mut work = [0.0f64; 8];
        let result = bdsqr::<f64, _, _, _>(
            Uplo::Lower,
            &mut d,
            &mut e,
            &mut vt,
            &mut u,
            &mut c,
            &mut work,
        );
        assert!(result.is_ok());
        // Singular values should be in descending order
        assert!(d[0] >= d[1]);
        assert!(d[0] > 0.0);
        assert!(d[1] > 0.0);
    }

    #[test]
    fn test_three_by_three_upper() {
        // 3x3 upper bidiagonal matrix:
        // [2 1 0]
        // [0 3 1]
        // [0 0 4]
        let mut d = [2.0f64, 3.0, 4.0];
        let mut e = [1.0f64, 1.0];
        let mut vt: Array2<f64> = Array2::zeros((3, 0));
        let mut u: Array2<f64> = Array2::zeros((0, 3));
        let mut c: Array2<f64> = Array2::zeros((3, 0));
        let mut work = [0.0f64; 12];
        let result = bdsqr::<f64, _, _, _>(
            Uplo::Upper,
            &mut d,
            &mut e,
            &mut vt,
            &mut u,
            &mut c,
            &mut work,
        );
        assert!(result.is_ok());
        // Singular values should be positive and sorted
        assert!(d[0] > 0.0);
        assert!(d[1] > 0.0);
        assert!(d[2] > 0.0);
        assert!(d[0] >= d[1]);
        assert!(d[1] >= d[2]);
    }

    #[test]
    fn test_diagonal_matrix() {
        // If all off-diagonal elements are zero, singular values are |d_i| sorted
        let mut d = [1.0f64, 4.0, 2.0, 3.0];
        let mut e = [0.0f64, 0.0, 0.0];
        let mut vt: Array2<f64> = Array2::zeros((4, 0));
        let mut u: Array2<f64> = Array2::zeros((0, 4));
        let mut c: Array2<f64> = Array2::zeros((4, 0));
        let mut work = [0.0f64; 16];
        let result = bdsqr::<f64, _, _, _>(
            Uplo::Upper,
            &mut d,
            &mut e,
            &mut vt,
            &mut u,
            &mut c,
            &mut work,
        );
        assert!(result.is_ok());
        // Should be sorted in descending order
        assert_abs_diff_eq!(d[0], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[1], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[2], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[3], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_with_right_singular_vectors() {
        // Test with VT (right singular vectors)
        let mut d = [3.0f64, 2.0];
        let mut e = [1.0f64];
        let mut vt = arr2(&[[1.0, 0.0], [0.0, 1.0]]); // Identity
        let mut u: Array2<f64> = Array2::zeros((0, 2));
        let mut c: Array2<f64> = Array2::zeros((2, 0));
        let mut work = [0.0f64; 8];
        let result = bdsqr::<f64, _, _, _>(
            Uplo::Upper,
            &mut d,
            &mut e,
            &mut vt,
            &mut u,
            &mut c,
            &mut work,
        );
        assert!(result.is_ok());
        // Singular values should be positive and sorted
        assert!(d[0] >= d[1]);
        // VT should have been modified
        // Check that rows are orthonormal
        let row0_norm = vt[(0, 0)] * vt[(0, 0)] + vt[(0, 1)] * vt[(0, 1)];
        let row1_norm = vt[(1, 0)] * vt[(1, 0)] + vt[(1, 1)] * vt[(1, 1)];
        assert_abs_diff_eq!(row0_norm, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(row1_norm, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_with_left_singular_vectors() {
        // Test with U (left singular vectors)
        let mut d = [3.0f64, 2.0];
        let mut e = [1.0f64];
        let mut vt: Array2<f64> = Array2::zeros((2, 0));
        let mut u = arr2(&[[1.0, 0.0], [0.0, 1.0]]); // Identity
        let mut c: Array2<f64> = Array2::zeros((2, 0));
        let mut work = [0.0f64; 8];
        let result = bdsqr::<f64, _, _, _>(
            Uplo::Upper,
            &mut d,
            &mut e,
            &mut vt,
            &mut u,
            &mut c,
            &mut work,
        );
        assert!(result.is_ok());
        // Singular values should be positive and sorted
        assert!(d[0] >= d[1]);
        // U should have been modified
        // Check that columns are orthonormal
        let col0_norm = u[(0, 0)] * u[(0, 0)] + u[(1, 0)] * u[(1, 0)];
        let col1_norm = u[(0, 1)] * u[(0, 1)] + u[(1, 1)] * u[(1, 1)];
        assert_abs_diff_eq!(col0_norm, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(col1_norm, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_complex_right_singular_vectors() {
        // Test with complex VT
        let mut d = [3.0f64, 2.0];
        let mut e = [1.0f64];
        let mut vt = arr2(&[
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ]);
        let mut u: Array2<Complex64> = Array2::zeros((0, 2));
        let mut c: Array2<Complex64> = Array2::zeros((2, 0));
        let mut work = [0.0f64; 8];
        let result = bdsqr::<Complex64, _, _, _>(
            Uplo::Upper,
            &mut d,
            &mut e,
            &mut vt,
            &mut u,
            &mut c,
            &mut work,
        );
        assert!(result.is_ok());
        // Singular values should be positive and sorted
        assert!(d[0] >= d[1]);
    }

    #[test]
    fn test_f32_support() {
        let mut d = [3.0f32, 2.0];
        let mut e = [1.0f32];
        let mut vt: Array2<f32> = Array2::zeros((2, 0));
        let mut u: Array2<f32> = Array2::zeros((0, 2));
        let mut c: Array2<f32> = Array2::zeros((2, 0));
        let mut work = [0.0f32; 8];
        let result = bdsqr::<f32, _, _, _>(
            Uplo::Upper,
            &mut d,
            &mut e,
            &mut vt,
            &mut u,
            &mut c,
            &mut work,
        );
        assert!(result.is_ok());
        assert!(d[0] >= d[1]);
    }

    #[test]
    fn test_verify_svd_decomposition() {
        // Verify the SVD decomposition: B = U * S * VT
        // For a 2x2 upper bidiagonal:
        // [3 1]
        // [0 2]
        let mut d = [3.0f64, 2.0];
        let mut e = [1.0f64];
        let mut vt = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let mut u = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let mut c: Array2<f64> = Array2::zeros((2, 0));
        let mut work = [0.0f64; 8];

        // Save original matrix
        let b_orig = arr2(&[[3.0, 1.0], [0.0, 2.0]]);

        let result = bdsqr::<f64, _, _, _>(
            Uplo::Upper,
            &mut d,
            &mut e,
            &mut vt,
            &mut u,
            &mut c,
            &mut work,
        );
        assert!(result.is_ok());

        // Reconstruct: B_reconstructed = U * diag(d) * VT
        let s = arr2(&[[d[0], 0.0], [0.0, d[1]]]);
        let us = u.dot(&s);
        let b_reconstructed = us.dot(&vt);

        // Check that reconstruction matches original
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(b_reconstructed[(i, j)], b_orig[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_larger_matrix() {
        // Test with a larger matrix
        let n = 5;
        let mut d = vec![5.0f64, 4.0, 3.0, 2.0, 1.0];
        let mut e = vec![0.5f64, 0.5, 0.5, 0.5];
        let mut vt: Array2<f64> = Array2::zeros((n, 0));
        let mut u: Array2<f64> = Array2::zeros((0, n));
        let mut c: Array2<f64> = Array2::zeros((n, 0));
        let mut work = vec![0.0f64; 4 * n];
        let result = bdsqr::<f64, _, _, _>(
            Uplo::Upper,
            &mut d,
            &mut e,
            &mut vt,
            &mut u,
            &mut c,
            &mut work,
        );
        assert!(result.is_ok());
        // Singular values should be positive and sorted
        for i in 0..n {
            assert!(d[i] > 0.0, "Singular value {} should be positive", i);
            if i > 0 {
                assert!(d[i - 1] >= d[i], "Singular values should be sorted");
            }
        }
    }

    #[test]
    fn test_identity_like() {
        // Bidiagonal with all 1s on diagonal, small off-diagonal
        let mut d = [1.0f64, 1.0, 1.0, 1.0];
        let mut e = [0.1f64, 0.1, 0.1];
        let mut vt: Array2<f64> = Array2::zeros((4, 0));
        let mut u: Array2<f64> = Array2::zeros((0, 4));
        let mut c: Array2<f64> = Array2::zeros((4, 0));
        let mut work = [0.0f64; 16];
        let result = bdsqr::<f64, _, _, _>(
            Uplo::Upper,
            &mut d,
            &mut e,
            &mut vt,
            &mut u,
            &mut c,
            &mut work,
        );
        assert!(result.is_ok());
        // Check singular values are positive and sorted
        for i in 0..4 {
            assert!(d[i] > 0.0, "Singular value {} should be positive", i);
            if i > 0 {
                assert!(d[i - 1] >= d[i], "Singular values should be sorted");
            }
        }
    }
}
