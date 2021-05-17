use num_traits::Float;
use std::ops::{AddAssign, MulAssign};

/// Computes an approximation to the smallest eigenvalue using values of d from
/// the previous transform.
///
/// # Panics
///
/// Panics if `z` has fewer than `4 * n0` elements.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn lasq4<A, const PP: usize>(
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

    let const_1 = A::from(9. / 16.).expect("valid conversion");
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
                    i4 -= 4;
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
                    i4 -= 4;
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
                    i4 -= 4;
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
                    i4 -= 4;
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

/// Computes one dqds transform in poing-pong form.
///
/// # Panics
///
/// Panics if `n0 - i0 <= 1`.
#[allow(dead_code)]
fn lasq5<A, const PP: usize>(
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
    assert!(n0 - i0 <= 1);

    let d_thresh = eps * (sigma + tau);
    if tau < d_thresh / (A::one() + A::one()) {
        tau = A::zero();
    }

    let mut j4 = 4 * i0 + PP - 3;
    let mut e_min = z[j4 + 3];
    let mut d = z[j4 - 1] - tau;
    let mut d_min = d;
    let d_min_2;
    let d_n;
    let d_nm1;
    let d_nm2;
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

    d_nm2 = d;
    d_min_2 = d_min;
    j4 = 4 * (n0 - 2) - PP;
    let mut j4_p2 = j4 + 2 * PP - 1;
    z[j4 - 3] = d_nm2 + z[j4_p2 - 1];
    z[j4 - 1] = z[j4_p2 + 1] * (z[j4_p2 - 1] / z[j4 - 3]);
    d_nm1 = z[j4_p2 + 1] * (d_nm2 / z[j4 - 3]) - tau;
    if d_nm1 < d_min {
        d_min = d_nm1;
    }

    let d_min_1 = d_min;
    j4 += 4;
    j4_p2 = j4 + 2 * PP - 1;
    z[j4 - 3] = d_nm1 + z[j4_p2 - 1];
    z[j4 - 1] = z[j4_p2 + 1] * (z[j4_p2 - 1] / z[j4 - 3]);
    d_n = z[j4_p2 + 1] * (d_nm1 / z[j4 - 3]) - tau;
    if d_n < d_min {
        d_min = d_n;
    }

    z[j4 + 1] = d_n;
    z[4 * n0 - PP] = e_min;
    (d_min, d_min_1, d_min_2, d_n, d_nm1, d_nm2)
}

/// Computes one dqd transform in ping-pong form.
///
/// # Panics
///
/// Panics if `n0 - i0 <= 1`.
#[allow(dead_code)]
fn lasq6<A, const PP: usize>(i0: usize, n0: usize, z: &mut [A]) -> (A, A, A, A, A, A)
where
    A: Float + MulAssign,
{
    assert!(PP <= 1, "`PP` must be either 0 or 1.");
    assert!(n0 - i0 <= 1);

    let mut j4 = 4 * i0 + PP - 3;
    let mut e_min = z[j4 + 3];
    let mut d = z[j4 - 1];
    let mut d_min = d;

    for j4 in (4 * i0..=4 * (n0 - 3)).step_by(4) {
        z[j4 - PP - 3] = d + z[j4 + PP - 2];
        if z[j4 - PP - 3] == A::zero() {
            z[j4 - PP - 1] = A::zero();
            d = z[j4 + PP];
            d_min = d;
            e_min = A::zero();
        } else if A::min_positive_value() * z[j4 + PP] < z[j4 - PP - 3]
            && A::min_positive_value() * z[j4 - PP - 3] < z[j4 + PP]
        {
            let tmp = z[j4 + PP] / z[j4 - PP - 3];
            z[j4 - PP - 1] = z[j4 + PP - 2] * tmp;
            d *= tmp;
        } else {
            z[j4 - PP - 1] = z[j4 + PP] * (z[j4 + PP - 2] / z[j4 - PP - 3]);
            d = z[j4 + PP] * (d / z[j4 - PP - 3]);
        }
        if d < d_min {
            d_min = d;
        }
        let e = z[j4 - PP - 1];
        if e < e_min {
            e_min = e;
        }
    }

    let d_nm2 = d;
    let d_min_2 = d_min;
    j4 = 4 * (n0 - 2) - PP;
    let mut j4_p2 = j4 + 2 * PP - 1;
    z[j4 - 3] = d_nm2 + z[j4_p2 - 1];
    let d_nm1;
    if z[j4 - 3] == A::zero() {
        z[j4 - 1] = A::zero();
        d_nm1 = z[j4_p2 + 1];
        d_min = d_nm1;
        e_min = A::zero();
    } else if A::min_positive_value() * z[j4_p2 + 1] < z[j4 - 3]
        && A::min_positive_value() * z[j4 - 3] < z[j4_p2 + 1]
    {
        let tmp = z[j4_p2 + 1] / z[j4 - 3];
        z[j4 - 1] = z[j4_p2 - 1] * tmp;
        d_nm1 = d_nm2 * tmp;
    } else {
        z[j4 - 1] = z[j4_p2 + 1] * (z[j4_p2 - 1] / z[j4 - 3]);
        d_nm1 = z[j4_p2 + 1] * (d_nm2 / z[j4 - 3]);
    }
    if d_nm1 < d_min {
        d_min = d_nm1;
    }

    let d_min_1 = d_min;
    j4 += 4;
    j4_p2 = j4 + 2 * PP - 1;
    z[j4 - 3] = d_nm1 + z[j4_p2 - 1];
    let d_n;
    if z[j4 - 3] == A::zero() {
        z[j4 - 1] = A::zero();
        d_n = z[j4_p2 + 1];
        d_min = d_n;
        e_min = A::zero();
    } else if A::min_positive_value() * z[j4_p2 + 1] < z[j4 - 3]
        && A::min_positive_value() * z[j4 - 3] < z[j4_p2 + 1]
    {
        let tmp = z[j4_p2 + 1] / z[j4 - 3];
        z[j4 - 1] = z[j4_p2 - 1] / tmp;
        d_n = d_nm1 * tmp;
    } else {
        z[j4 - 1] = z[j4_p2 + 1] * (z[j4_p2 - 1] / z[j4 - 3]);
        d_n = z[j4_p2 + 1] / (d_nm1 / z[j4 - 3]);
    }
    if d_n < d_min {
        d_min = d_n;
    }

    z[j4 + 1] = d_n;
    z[4 * n0 - PP - 1] = e_min;
    (d_min, d_min_1, d_min_2, d_n, d_nm1, d_nm2)
}
