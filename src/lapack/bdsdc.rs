//! Bidiagonal SVD using divide and conquer.
//!
//! This module provides the `bdsdc` function which computes the singular value
//! decomposition of a real bidiagonal matrix using a divide-and-conquer algorithm.

#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::needless_range_loop)]

use ndarray::{Array2, ArrayBase, DataMut, Ix2};
use num_traits::FromPrimitive;

use super::lartg::lartg;
use super::las2::las2;
use super::lasv2::lasv2;
use crate::Real;

/// Specifies whether singular vectors are computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CompQ {
    /// Compute singular values only.
    None,
    /// Compute singular values and singular vectors in compact form.
    #[allow(dead_code)]
    Compact,
    /// Compute singular values and full singular vectors.
    Identity,
}

/// Specifies whether the bidiagonal matrix is upper or lower bidiagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Uplo {
    /// Upper bidiagonal: has nonzero elements on diagonal and superdiagonal
    Upper,
    /// Lower bidiagonal: has nonzero elements on diagonal and subdiagonal
    #[allow(dead_code)]
    Lower,
}

/// Computes the singular value decomposition of a real bidiagonal matrix
/// using a divide and conquer method.
///
/// `bdsdc` computes the singular value decomposition (SVD) of a real
/// N-by-N (upper or lower) bidiagonal matrix B: B = U * S * VT,
/// using a divide and conquer method.
///
/// # Arguments
///
/// * `uplo` - Specifies whether B is upper or lower bidiagonal.
/// * `compq` - Specifies whether singular vectors are computed.
/// * `d` - On entry, the N diagonal elements of the bidiagonal matrix B.
///   On exit, if Ok, the singular values of B in decreasing order.
/// * `e` - On entry, the N-1 off-diagonal elements of the bidiagonal matrix B.
///   On exit, the contents are overwritten.
/// * `u` - On exit, if compq is Identity, contains the left singular vectors.
///   Must be N-by-N.
/// * `vt` - On exit, if compq is Identity, contains the right singular vectors.
///   Must be N-by-N.
/// * `work` - Workspace array. Size should be at least 4*N.
///
/// # Returns
///
/// * `Ok(())` if successful.
/// * `Err(info)` where `info > 0` means the algorithm did not converge.
#[allow(dead_code)]
#[allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]
pub fn bdsdc<A, SU, SVT>(
    uplo: Uplo,
    compq: CompQ,
    d: &mut [A],
    e: &mut [A],
    u: &mut ArrayBase<SU, Ix2>,
    vt: &mut ArrayBase<SVT, Ix2>,
    _work: &mut [A],
) -> Result<(), usize>
where
    A: Real + FromPrimitive,
    SU: DataMut<Elem = A>,
    SVT: DataMut<Elem = A>,
{
    let n = d.len();
    let zero = A::zero();
    let one = A::one();

    // Quick return if possible
    if n == 0 {
        return Ok(());
    }

    // Initialize U and VT to identity if computing singular vectors
    if compq == CompQ::Identity {
        for i in 0..n {
            for j in 0..n {
                u[(i, j)] = if i == j { one } else { zero };
                vt[(i, j)] = if i == j { one } else { zero };
            }
        }
    }

    if n == 1 {
        if d[0] < zero {
            d[0] = -d[0];
            if compq == CompQ::Identity {
                vt[(0, 0)] = -one;
            }
        }
        return Ok(());
    }

    // If lower bidiagonal, convert to upper bidiagonal
    if uplo == Uplo::Lower {
        for i in 0..n - 1 {
            let (cs, sn, r) = lartg(d[i], e[i]);
            d[i] = r;
            e[i] = sn * d[i + 1];
            d[i + 1] = cs * d[i + 1];

            if compq == CompQ::Identity {
                // Apply rotation to U from the right
                for k in 0..n {
                    let temp = u[(k, i + 1)];
                    u[(k, i + 1)] = cs * temp - sn * u[(k, i)];
                    u[(k, i)] = sn * temp + cs * u[(k, i)];
                }
            }
        }
    }

    // Use divide and conquer algorithm
    divide_and_conquer(compq, d, e, u, vt, 0, n)
}

/// Divide and conquer algorithm for bidiagonal SVD.
#[allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::too_many_arguments
)]
fn divide_and_conquer<A, SU, SVT>(
    compq: CompQ,
    d: &mut [A],
    e: &mut [A],
    u: &mut ArrayBase<SU, Ix2>,
    vt: &mut ArrayBase<SVT, Ix2>,
    start: usize,
    end: usize,
) -> Result<(), usize>
where
    A: Real + FromPrimitive,
    SU: DataMut<Elem = A>,
    SVT: DataMut<Elem = A>,
{
    let n = end - start;
    let zero = A::zero();
    let _one = A::one();

    // Base case: small matrices use direct QR iteration
    let smlsiz: usize = 25;
    if n <= smlsiz {
        return qr_svd(compq, d, e, u, vt, start, end);
    }

    // Divide: split at the middle
    let mid = start + n / 2;

    // Save the connection point values
    let alpha = d[mid];
    let beta = if mid > start { e[mid - 1] } else { zero };

    // Deflate: set e[mid-1] to zero
    if mid > start {
        e[mid - 1] = zero;
    }

    // Recursively solve the two subproblems
    divide_and_conquer(compq, d, e, u, vt, start, mid)?;
    divide_and_conquer(compq, d, e, u, vt, mid, end)?;

    // Conquer: merge the two subproblems
    merge_subproblems(compq, d, u, vt, start, mid, end, alpha, beta)
}

/// QR iteration for small bidiagonal matrices.
#[allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]
fn qr_svd<A, SU, SVT>(
    compq: CompQ,
    d: &mut [A],
    e: &mut [A],
    u: &mut ArrayBase<SU, Ix2>,
    vt: &mut ArrayBase<SVT, Ix2>,
    start: usize,
    end: usize,
) -> Result<(), usize>
where
    A: Real + FromPrimitive,
    SU: DataMut<Elem = A>,
    SVT: DataMut<Elem = A>,
{
    let n = end - start;
    let zero = A::zero();
    let one = A::one();

    if n == 0 {
        return Ok(());
    }

    if n == 1 {
        if d[start] < zero {
            d[start] = -d[start];
            if compq == CompQ::Identity {
                for j in 0..vt.ncols() {
                    vt[(start, j)] = -vt[(start, j)];
                }
            }
        }
        return Ok(());
    }

    if n == 2 {
        // Use lasv2 for 2x2 case
        let (sigmn, sigmx, sinr, cosr, sinl, cosl) = lasv2(d[start], e[start], d[start + 1]);
        d[start] = sigmx;
        d[start + 1] = sigmn;
        e[start] = zero;

        if compq == CompQ::Identity {
            // Update VT: VT = R * VT
            let ncols = vt.ncols();
            for j in 0..ncols {
                let temp = vt[(start, j)] * cosr + vt[(start + 1, j)] * sinr;
                vt[(start + 1, j)] = vt[(start + 1, j)] * cosr - vt[(start, j)] * sinr;
                vt[(start, j)] = temp;
            }
            // Update U: U = U * L^T
            let nrows = u.nrows();
            for i in 0..nrows {
                let temp = u[(i, start)] * cosl + u[(i, start + 1)] * sinl;
                u[(i, start + 1)] = u[(i, start + 1)] * cosl - u[(i, start)] * sinl;
                u[(i, start)] = temp;
            }
        }

        return Ok(());
    }

    // For larger submatrices, use implicit zero-shift QR iteration
    let rotate = compq == CompQ::Identity;
    let eps = A::epsilon();
    let unfl = A::min_positive_value();
    let ten: A = FromPrimitive::from_f64(10.0).unwrap();
    let hndrd: A = FromPrimitive::from_f64(100.0).unwrap();
    let meigth: A = FromPrimitive::from_f64(-0.125).unwrap();

    let tolmul = ten.max(hndrd.min(eps.powf(meigth)));
    let tol = tolmul * eps;

    // Compute approximate maximum singular value
    let mut smax = zero;
    for i in start..end {
        smax = smax.max(d[i].abs());
    }
    for i in start..end - 1 {
        smax = smax.max(e[i].abs());
    }

    let n_real: A = FromPrimitive::from_usize(n).unwrap();
    let maxitr: A = FromPrimitive::from_f64(6.0).unwrap();
    let thresh = maxitr * (n_real * (n_real * unfl));

    let maxitdivn = (maxitr * n_real).to_usize().unwrap();
    let mut iterdivn = 0;
    let mut iter = 0_isize;
    let mut m = end;
    let mut oldll = usize::MAX;
    let mut oldm = usize::MAX;

    loop {
        if m <= start + 1 {
            break;
        }

        if iter >= n as isize {
            iter -= n as isize;
            iterdivn += 1;
            if iterdivn >= maxitdivn {
                let mut info = 0;
                for i in start..end - 1 {
                    if e[i] != zero {
                        info += 1;
                    }
                }
                return Err(info);
            }
        }

        // Find diagonal block
        if tol < zero && d[m - 1].abs() <= thresh {
            d[m - 1] = zero;
        }

        let mut ll = start;
        smax = d[m - 1].abs();

        let mut found_split = false;
        for lll in 1..(m - start) {
            let idx = m - lll - 1;
            if idx < start {
                break;
            }
            let abss = d[idx].abs();
            let abse = e[idx].abs();
            if tol < zero && abss <= thresh {
                d[idx] = zero;
            }
            if abse <= thresh {
                found_split = true;
                ll = idx;
                break;
            }
            smax = smax.max(abss).max(abse);
        }

        if found_split {
            e[ll] = zero;
            if ll == m - 2 {
                m -= 1;
                continue;
            }
            ll += 1;
        } else {
            ll = start;
        }

        if ll == m - 2 {
            // 2x2 block
            let (sigmn, sigmx, sinr, cosr, sinl, cosl) = lasv2(d[m - 2], e[m - 2], d[m - 1]);
            d[m - 2] = sigmx;
            e[m - 2] = zero;
            d[m - 1] = sigmn;

            if rotate {
                let ncols = vt.ncols();
                for j in 0..ncols {
                    let temp = vt[(m - 2, j)] * cosr + vt[(m - 1, j)] * sinr;
                    vt[(m - 1, j)] = vt[(m - 1, j)] * cosr - vt[(m - 2, j)] * sinr;
                    vt[(m - 2, j)] = temp;
                }
                let nrows = u.nrows();
                for i in 0..nrows {
                    let temp = u[(i, m - 2)] * cosl + u[(i, m - 1)] * sinl;
                    u[(i, m - 1)] = u[(i, m - 1)] * cosl - u[(i, m - 2)] * sinl;
                    u[(i, m - 2)] = temp;
                }
            }
            m -= 2;
            continue;
        }

        // Choose shift direction
        let idir = if ll > oldm || m < oldll {
            if d[ll].abs() >= d[m - 1].abs() {
                1
            } else {
                2
            }
        } else if d[ll].abs() >= d[m - 1].abs() {
            1
        } else {
            2
        };

        // Apply convergence tests
        let mut smin = zero;
        if idir == 1 {
            if e[m - 2].abs() <= tol.abs() * d[m - 1].abs()
                || (tol < zero && e[m - 2].abs() <= thresh)
            {
                e[m - 2] = zero;
                continue;
            }

            if tol >= zero {
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
            if e[ll].abs() <= tol.abs() * d[ll].abs() || (tol < zero && e[ll].abs() <= thresh) {
                e[ll] = zero;
                continue;
            }

            if tol >= zero {
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

        // Compute shift
        let hndrth: A = FromPrimitive::from_f64(0.01).unwrap();
        let shift = if tol >= zero && n_real * tol * (smin / smax) <= eps.max(hndrth * tol) {
            zero
        } else {
            let (shift_val, _) = if idir == 1 {
                las2(d[m - 2], e[m - 2], d[m - 1])
            } else {
                las2(d[ll], e[ll], d[ll + 1])
            };

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

        iter += (m - ll) as isize;

        // QR iteration step
        if shift == zero {
            // Zero shift
            if idir == 1 {
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

                    if rotate {
                        // Apply rotation to VT from the left
                        let ncols = vt.ncols();
                        for j in 0..ncols {
                            let temp = vt[(i, j)] * cs_new + vt[(i + 1, j)] * sn;
                            vt[(i + 1, j)] = vt[(i + 1, j)] * cs_new - vt[(i, j)] * sn;
                            vt[(i, j)] = temp;
                        }
                        // Apply rotation to U from the right
                        let nrows = u.nrows();
                        for k in 0..nrows {
                            let temp = u[(k, i)] * oldcs_new + u[(k, i + 1)] * oldsn_new;
                            u[(k, i + 1)] = u[(k, i + 1)] * oldcs_new - u[(k, i)] * oldsn_new;
                            u[(k, i)] = temp;
                        }
                    }
                }
                let h = d[m - 1] * cs;
                d[m - 1] = h * oldcs;
                e[m - 2] = h * oldsn;

                if e[m - 2].abs() <= thresh {
                    e[m - 2] = zero;
                }
            } else {
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

                    if rotate {
                        let ncols = vt.ncols();
                        for j in 0..ncols {
                            let temp = vt[(i, j)] * cs_new + vt[(i - 1, j)] * sn;
                            vt[(i - 1, j)] = vt[(i - 1, j)] * cs_new - vt[(i, j)] * sn;
                            vt[(i, j)] = temp;
                        }
                        let nrows = u.nrows();
                        for k in 0..nrows {
                            let temp = u[(k, i)] * oldcs_new + u[(k, i - 1)] * oldsn_new;
                            u[(k, i - 1)] = u[(k, i - 1)] * oldcs_new - u[(k, i)] * oldsn_new;
                            u[(k, i)] = temp;
                        }
                    }
                }
                let h = d[ll] * cs;
                d[ll] = h * oldcs;
                e[ll] = h * oldsn;

                if e[ll].abs() <= thresh {
                    e[ll] = zero;
                }
            }
        } else {
            // Nonzero shift
            if idir == 1 {
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

                    if rotate {
                        let ncols = vt.ncols();
                        for j in 0..ncols {
                            let temp = vt[(i, j)] * cosr + vt[(i + 1, j)] * sinr;
                            vt[(i + 1, j)] = vt[(i + 1, j)] * cosr - vt[(i, j)] * sinr;
                            vt[(i, j)] = temp;
                        }
                    }

                    let (cosl, sinl, r) = lartg(f, g);
                    d[i] = r;
                    f = cosl * e[i] + sinl * d[i + 1];
                    d[i + 1] = cosl * d[i + 1] - sinl * e[i];
                    if i < m - 2 {
                        g = sinl * e[i + 1];
                        e[i + 1] = cosl * e[i + 1];
                    }

                    if rotate {
                        let nrows = u.nrows();
                        for k in 0..nrows {
                            let temp = u[(k, i)] * cosl + u[(k, i + 1)] * sinl;
                            u[(k, i + 1)] = u[(k, i + 1)] * cosl - u[(k, i)] * sinl;
                            u[(k, i)] = temp;
                        }
                    }
                }
                e[m - 2] = f;

                if e[m - 2].abs() <= thresh {
                    e[m - 2] = zero;
                }
            } else {
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

                    if rotate {
                        let ncols = vt.ncols();
                        for j in 0..ncols {
                            let temp = vt[(i, j)] * cosr + vt[(i - 1, j)] * sinr;
                            vt[(i - 1, j)] = vt[(i - 1, j)] * cosr - vt[(i, j)] * sinr;
                            vt[(i, j)] = temp;
                        }
                    }

                    let (cosl, sinl, r) = lartg(f, g);
                    d[i] = r;
                    f = cosl * e[i - 1] + sinl * d[i - 1];
                    d[i - 1] = cosl * d[i - 1] - sinl * e[i - 1];
                    if i > ll + 1 {
                        g = sinl * e[i - 2];
                        e[i - 2] = cosl * e[i - 2];
                    }

                    if rotate {
                        let nrows = u.nrows();
                        for k in 0..nrows {
                            let temp = u[(k, i)] * cosl + u[(k, i - 1)] * sinl;
                            u[(k, i - 1)] = u[(k, i - 1)] * cosl - u[(k, i)] * sinl;
                            u[(k, i)] = temp;
                        }
                    }
                }
                e[ll] = f;

                if e[ll].abs() <= thresh {
                    e[ll] = zero;
                }
            }
        }
    }

    // Make singular values positive
    for i in start..end {
        if d[i] < zero {
            d[i] = -d[i];
            if rotate {
                for j in 0..vt.ncols() {
                    vt[(i, j)] = -vt[(i, j)];
                }
            }
        }
    }

    // Sort singular values in decreasing order (within the subblock)
    for i in start..end - 1 {
        let mut k = i;
        let mut p = d[i];
        for j in i + 1..end {
            if d[j] > p {
                k = j;
                p = d[j];
            }
        }
        if k != i {
            d[k] = d[i];
            d[i] = p;
            if rotate {
                // Swap columns of U
                for r in 0..u.nrows() {
                    let temp = u[(r, i)];
                    u[(r, i)] = u[(r, k)];
                    u[(r, k)] = temp;
                }
                // Swap rows of VT
                for c in 0..vt.ncols() {
                    let temp = vt[(i, c)];
                    vt[(i, c)] = vt[(k, c)];
                    vt[(k, c)] = temp;
                }
            }
        }
    }

    Ok(())
}

/// Merge two subproblems after divide step.
#[allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]
fn merge_subproblems<A, SU, SVT>(
    compq: CompQ,
    d: &mut [A],
    u: &mut ArrayBase<SU, Ix2>,
    vt: &mut ArrayBase<SVT, Ix2>,
    start: usize,
    mid: usize,
    end: usize,
    alpha: A,
    beta: A,
) -> Result<(), usize>
where
    A: Real + FromPrimitive,
    SU: DataMut<Elem = A>,
    SVT: DataMut<Elem = A>,
{
    let n = end - start;
    let zero = A::zero();
    let one = A::one();

    // Collect singular values from both subproblems
    let mut sigma: Vec<A> = Vec::with_capacity(n);
    for i in start..end {
        sigma.push(d[i]);
    }

    // Construct the z vector for the secular equation
    // z[i] = u_i^T * [alpha, 0, ..., 0, beta, 0, ...]^T
    let mut z: Vec<A> = vec![zero; n];
    if compq == CompQ::Identity {
        // z is constructed from the last row of U1 (scaled by alpha)
        // and the first row of VT2 (scaled by beta)
        for i in 0..n {
            if i < mid - start {
                // From U1: last row, column i
                z[i] = alpha * u[(mid - 1, start + i)];
            } else {
                // From VT2: first row, column (i - (mid - start))
                z[i] = beta * vt[(mid, start + i)];
            }
        }
    } else {
        // Simplified case without computing vectors
        if mid > start && mid - 1 - start < n {
            z[mid - 1 - start] = alpha;
        }
        if mid - start < n {
            z[mid - start] = beta;
        }
    }

    // Sort the singular values and keep track of permutation
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by(|&a, &b| sigma[b].partial_cmp(&sigma[a]).unwrap());

    let sorted_sigma: Vec<A> = perm.iter().map(|&i| sigma[i]).collect();
    let sorted_z: Vec<A> = perm.iter().map(|&i| z[i]).collect();

    // Find new singular values using secular equation solver
    let mut new_sigma: Vec<A> = vec![zero; n];
    for k in 0..n {
        new_sigma[k] = solve_secular_equation(&sorted_sigma, &sorted_z, k);
    }

    // Sort new singular values in decreasing order
    new_sigma.sort_by(|a, b| b.partial_cmp(a).unwrap());

    // Copy back to d
    d[start..(n + start)].copy_from_slice(&new_sigma[..n]);

    // Update singular vectors if needed
    if compq == CompQ::Identity {
        let full_n = u.nrows();
        let full_m = vt.ncols();

        // Compute new singular vectors
        let mut u_new: Array2<A> = Array2::zeros((n, n));
        let mut vt_new: Array2<A> = Array2::zeros((n, n));

        for k in 0..n {
            let sigma_k = new_sigma[k];
            let sigma_k_sq = sigma_k * sigma_k;

            // Compute eigenvector for secular equation
            let mut norm_sq = zero;
            for i in 0..n {
                let old_sigma_sq = sorted_sigma[i] * sorted_sigma[i];
                let diff = old_sigma_sq - sigma_k_sq;
                if diff.abs() > A::epsilon() * (old_sigma_sq + sigma_k_sq + one) {
                    let val = sorted_z[i] / diff;
                    u_new[(perm[i], k)] = val;
                    vt_new[(k, perm[i])] = val;
                    norm_sq += val * val;
                } else {
                    u_new[(perm[i], k)] = one;
                    vt_new[(k, perm[i])] = one;
                    norm_sq += one;
                }
            }

            // Normalize
            let norm = norm_sq.sqrt();
            if norm > zero {
                for i in 0..n {
                    u_new[(i, k)] /= norm;
                    vt_new[(k, i)] /= norm;
                }
            }
        }

        // Apply: U_new_full = U[:, start:end] * u_new
        let mut temp_u: Array2<A> = Array2::zeros((full_n, n));
        for i in 0..full_n {
            for j in 0..n {
                let mut sum = zero;
                for k in 0..n {
                    sum += u[(i, start + k)] * u_new[(k, j)];
                }
                temp_u[(i, j)] = sum;
            }
        }
        for i in 0..full_n {
            for j in 0..n {
                u[(i, start + j)] = temp_u[(i, j)];
            }
        }

        // Apply: VT_new_full = vt_new * VT[start:end, :]
        let mut temp_vt: Array2<A> = Array2::zeros((n, full_m));
        for i in 0..n {
            for j in 0..full_m {
                let mut sum = zero;
                for k in 0..n {
                    sum += vt_new[(i, k)] * vt[(start + k, j)];
                }
                temp_vt[(i, j)] = sum;
            }
        }
        for i in 0..n {
            for j in 0..full_m {
                vt[(start + i, j)] = temp_vt[(i, j)];
            }
        }
    }

    Ok(())
}

/// Solve the secular equation to find the k-th singular value.
#[allow(clippy::many_single_char_names)]
fn solve_secular_equation<A: Real + FromPrimitive>(sigma: &[A], z: &[A], k: usize) -> A {
    let n = sigma.len();
    let zero = A::zero();
    let one = A::one();
    let two = one + one;

    if n == 0 {
        return zero;
    }

    // The secular equation is:
    // f(x) = 1 + sum_i z_i^2 / (sigma_i^2 - x^2) = 0
    // The k-th root lies in the interval (sigma[k+1], sigma[k])

    let upper = if k < n { sigma[k] } else { zero };
    let lower = if k + 1 < n { sigma[k + 1] } else { zero };

    // Handle case where upper == lower
    if (upper - lower).abs() < A::epsilon() * upper.max(one) {
        return upper;
    }

    // Use bisection with Newton refinement
    let mut x = (upper + lower) / two;
    let max_iter = 50;
    let tol = A::epsilon() * upper.max(one);

    for _ in 0..max_iter {
        let x_sq = x * x;

        // Compute f(x) and f'(x)
        let mut f = one;
        let mut fp = zero;

        for i in 0..n {
            let sigma_sq = sigma[i] * sigma[i];
            let denom = sigma_sq - x_sq;
            if denom.abs() > tol {
                let z_sq = z[i] * z[i];
                f += z_sq / denom;
                fp += two * x * z_sq / (denom * denom);
            }
        }

        // Newton step
        if fp.abs() > tol {
            let dx = f / fp;
            let new_x = x - dx;

            // Ensure x stays in bounds
            x = if new_x <= lower {
                (lower + x) / two
            } else if new_x >= upper {
                (upper + x) / two
            } else {
                new_x
            };

            if dx.abs() < tol {
                break;
            }
        } else {
            break;
        }
    }

    x
}

/// Returns the absolute value of a with the sign of b.
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

    use super::*;

    #[test]
    fn test_empty() {
        let mut d: [f64; 0] = [];
        let mut e: [f64; 0] = [];
        let mut u: Array2<f64> = Array2::zeros((0, 0));
        let mut vt: Array2<f64> = Array2::zeros((0, 0));
        let mut work: [f64; 0] = [];
        let result = bdsdc(
            Uplo::Upper,
            CompQ::None,
            &mut d,
            &mut e,
            &mut u,
            &mut vt,
            &mut work,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_single_element() {
        let mut d = [3.0f64];
        let mut e: [f64; 0] = [];
        let mut u: Array2<f64> = Array2::zeros((1, 1));
        let mut vt: Array2<f64> = Array2::zeros((1, 1));
        let mut work = [0.0f64; 4];
        let result = bdsdc(
            Uplo::Upper,
            CompQ::Identity,
            &mut d,
            &mut e,
            &mut u,
            &mut vt,
            &mut work,
        );
        assert!(result.is_ok());
        assert_eq!(d[0], 3.0);
        assert_eq!(u[(0, 0)], 1.0);
        assert_eq!(vt[(0, 0)], 1.0);
    }

    #[test]
    fn test_single_element_negative() {
        let mut d = [-5.0f64];
        let mut e: [f64; 0] = [];
        let mut u: Array2<f64> = Array2::zeros((1, 1));
        let mut vt: Array2<f64> = Array2::zeros((1, 1));
        let mut work = [0.0f64; 4];
        let result = bdsdc(
            Uplo::Upper,
            CompQ::Identity,
            &mut d,
            &mut e,
            &mut u,
            &mut vt,
            &mut work,
        );
        assert!(result.is_ok());
        assert_eq!(d[0], 5.0);
    }

    #[test]
    fn test_two_by_two_upper() {
        let mut d = [3.0f64, 2.0];
        let mut e = [1.0f64];
        let mut u: Array2<f64> = Array2::zeros((2, 2));
        let mut vt: Array2<f64> = Array2::zeros((2, 2));
        let mut work = [0.0f64; 16];
        let result = bdsdc(
            Uplo::Upper,
            CompQ::Identity,
            &mut d,
            &mut e,
            &mut u,
            &mut vt,
            &mut work,
        );
        assert!(result.is_ok());
        assert!(d[0] >= d[1]);
        assert!(d[0] > 0.0);
        assert!(d[1] > 0.0);
    }

    #[test]
    fn test_diagonal_matrix() {
        let mut d = [1.0f64, 4.0, 2.0, 3.0];
        let mut e = [0.0f64, 0.0, 0.0];
        let mut u: Array2<f64> = Array2::zeros((4, 4));
        let mut vt: Array2<f64> = Array2::zeros((4, 4));
        let mut work = [0.0f64; 64];
        let result = bdsdc(
            Uplo::Upper,
            CompQ::Identity,
            &mut d,
            &mut e,
            &mut u,
            &mut vt,
            &mut work,
        );
        assert!(result.is_ok());
        assert_abs_diff_eq!(d[0], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[1], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[2], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[3], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_no_vectors() {
        let mut d = [3.0f64, 2.0];
        let mut e = [1.0f64];
        let mut u: Array2<f64> = Array2::zeros((0, 0));
        let mut vt: Array2<f64> = Array2::zeros((0, 0));
        let mut work = [0.0f64; 8];
        let result = bdsdc(
            Uplo::Upper,
            CompQ::None,
            &mut d,
            &mut e,
            &mut u,
            &mut vt,
            &mut work,
        );
        assert!(result.is_ok());
        assert!(d[0] >= d[1]);
    }

    #[test]
    fn test_verify_decomposition() {
        // Verify B = U * diag(d) * VT for a 2x2 matrix
        let mut d = [3.0f64, 2.0];
        let mut e = [1.0f64];
        let mut u: Array2<f64> = Array2::zeros((2, 2));
        let mut vt: Array2<f64> = Array2::zeros((2, 2));
        let mut work = [0.0f64; 16];

        // Original bidiagonal matrix
        let b_orig = arr2(&[[3.0, 1.0], [0.0, 2.0]]);

        let result = bdsdc(
            Uplo::Upper,
            CompQ::Identity,
            &mut d,
            &mut e,
            &mut u,
            &mut vt,
            &mut work,
        );
        assert!(result.is_ok());

        // Reconstruct: B = U * diag(d) * VT
        let s = arr2(&[[d[0], 0.0], [0.0, d[1]]]);
        let b_reconstructed = u.dot(&s).dot(&vt);

        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(b_reconstructed[(i, j)], b_orig[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_f32_support() {
        let mut d = [3.0f32, 2.0];
        let mut e = [1.0f32];
        let mut u: Array2<f32> = Array2::zeros((2, 2));
        let mut vt: Array2<f32> = Array2::zeros((2, 2));
        let mut work = [0.0f32; 16];
        let result = bdsdc(
            Uplo::Upper,
            CompQ::Identity,
            &mut d,
            &mut e,
            &mut u,
            &mut vt,
            &mut work,
        );
        assert!(result.is_ok());
        assert!(d[0] >= d[1]);
    }

    #[test]
    fn test_three_by_three() {
        let mut d = [2.0f64, 3.0, 4.0];
        let mut e = [1.0f64, 1.0];
        let mut u: Array2<f64> = Array2::zeros((3, 3));
        let mut vt: Array2<f64> = Array2::zeros((3, 3));
        let mut work = [0.0f64; 48];
        let result = bdsdc(
            Uplo::Upper,
            CompQ::Identity,
            &mut d,
            &mut e,
            &mut u,
            &mut vt,
            &mut work,
        );
        assert!(result.is_ok());
        assert!(d[0] > 0.0);
        assert!(d[1] > 0.0);
        assert!(d[2] > 0.0);
        assert!(d[0] >= d[1]);
        assert!(d[1] >= d[2]);
    }
}
