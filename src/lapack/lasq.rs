use crate::Real;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum PingPong {
    Ping,
    Pong,
}

/// Computes an approximation to the smallest eigenvalue using values of d from
/// the previous transform.
#[allow(dead_code)]
#[allow(
    clippy::cast_possible_wrap,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
pub(crate) unsafe fn lasq4<T: Real>(
    i0: usize,
    n0: usize,
    z: *mut T,
    pp: PingPong,
    n0_in: usize,
    d_min: T,
    d_min_1: T,
    d_min_2: T,
    d_n: T,
    d_n_1: T,
    d_n_2: T,
    mut g: T,
    mut ttype: isize,
) -> (T, isize, T) {
    if d_min <= T::zero() {
        return (-d_min, -1, g);
    }
    let pong = if let PingPong::Pong = pp { 1 } else { 0 };

    let const_1 = T::from(9. / 16.).expect("valid conversion");
    let const_2 = T::from(1.01).expect("valid conversion");
    let const_3 = T::from(1.05).expect("valid conversion");
    let hundred = T::from(100).expect("valid conversion");
    let half = (T::one() + T::one()).recip();
    let third = (T::one() + T::one() + T::one()).recip();
    let quarter = (T::one() + T::one() + T::one() + T::one()).recip();

    let nn = 4 * n0 as isize + pong;
    let mut s = T::zero();
    if n0_in == n0 {
        if d_min == d_n || d_min == d_n_1 {
            let mut b_1 = (*z.offset(nn - 3)).sqrt() * (*z.offset(nn - 5)).sqrt();

            if d_min == d_n && d_min_1 == d_n_1 {
                let b_2 = (*z.offset(nn - 7)).sqrt() * (*z.offset(nn - 9)).sqrt();
                let a_2 = *z.offset(nn - 7) + *z.offset(nn - 5);
                let gap_2 = d_min_2 - a_2 - d_min_2 * quarter;
                let gap_1 = if gap_2 > T::zero() && gap_2 > b_2 {
                    a_2 - d_n - (b_2 / gap_2) * b_2
                } else {
                    a_2 - d_n - (b_1 + b_2)
                };
                if gap_1 > T::zero() && gap_1 > b_1 {
                    let left = d_n - (b_1 / gap_1) * b_1;
                    let right = d_min * half;
                    s = if left > right { left } else { right };
                    ttype = -2;
                } else {
                    s = if d_n > b_1 { d_n - b_1 } else { T::zero() };
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
                    a_2 = T::zero();
                    if *z.offset(nn - 5) > *z.offset(nn - 7) {
                        return (s, ttype, g);
                    }
                    b_2 = *z.offset(nn - 5) / *z.offset(nn - 7);
                    np = nn - 9;
                } else {
                    np = nn - pong * 2;
                    gam = d_n_1;
                    if *z.offset(np - 4) > *z.offset(np - 2) {
                        return (s, ttype, g);
                    }
                    a_2 = *z.offset(np - 4) / *z.offset(np - 2);
                    if *z.offset(nn - 9) > *z.offset(nn - 11) {
                        return (s, ttype, g);
                    }
                    b_2 = *z.offset(nn - 9) / *z.offset(nn - 11);
                    np = nn - 13;
                }

                a_2 += b_2;
                let mut i4 = np;
                while i4 >= 4 * i0 as isize - 1 + pong {
                    if b_2 == T::zero() {
                        break;
                    }
                    b_1 = b_2;
                    if *z.offset(i4) > *z.offset(i4 - 2) {
                        return (s, ttype, g);
                    }
                    b_2 *= *z.offset(i4) / *z.offset(i4 - 2);
                    a_2 += b_2;
                    let max_b = if b_2 > b_1 { b_2 } else { b_1 };
                    if hundred * max_b < a_2 || const_1 < a_2 {
                        break;
                    }
                    i4 -= 4;
                }
                a_2 *= const_3;

                if a_2 < const_1 {
                    s = gam * (T::one() - a_2.sqrt()) / (T::one() + a_2);
                }
            }
        } else if d_min == d_n_2 {
            ttype = -5;
            s = d_min * quarter;

            let np = nn - 2 * pong;
            let mut b_1 = *z.offset(np - 2);
            let mut b_2 = *z.offset(np - 6);
            let gam = d_n_2;
            if *z.offset(np - 8) > b_2 || *z.offset(np - 4) > b_1 {
                return (s, ttype, g);
            }
            let mut a_2 = (*z.offset(np - 8) / b_2) * (T::one() + *z.offset(np - 4) / b_1);

            if n0 as isize - i0 as isize > 2 {
                b_2 = *z.offset(nn - 13) / *z.offset(nn - 15);
                a_2 += b_2;
                let mut i4 = nn - 17;
                while i4 >= 4 * i0 as isize - 1 + pong {
                    if b_2 == T::zero() {
                        break;
                    }
                    b_1 = b_2;
                    if *z.offset(i4) > *z.offset(i4 - 2) {
                        return (s, ttype, g);
                    }
                    b_2 *= *z.offset(i4) / *z.offset(i4 - 2);
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
                s = gam * (T::one() - a_2.sqrt()) / (T::one() + a_2);
            }
        } else {
            if ttype == -6 {
                g += (T::one() - g) * third;
            } else if ttype == -18 {
                g = quarter * third;
            } else {
                g = quarter;
            }
            s = g * d_min;
            ttype = -6;
        }
    } else if n0_in == n0 + 1 {
        if d_min_1 == d_n && d_min_2 == d_n_2 {
            ttype = -7;
            s = third * d_min_1;
            if *z.offset(nn - 5) > *z.offset(nn - 7) {
                return (s, ttype, g);
            }
            let mut b_1 = *z.offset(nn - 5) / *z.offset(nn - 7);
            let mut b_2 = b_1;
            if b_2 != T::zero() {
                let mut i4 = 4 * n0 as isize - 9 + pong;
                while i4 >= 4 * i0 as isize - 1 + pong {
                    let a_2 = b_1;
                    if *z.offset(i4) > *z.offset(i4 - 2) {
                        return (s, ttype, g);
                    }
                    b_1 *= *z.offset(i4) / *z.offset(i4 - 2);
                    b_2 += b_1;
                    let ab_max = if b_1 > a_2 { b_1 } else { a_2 };
                    if hundred * ab_max < b_2 {
                        break;
                    }
                    i4 -= 4;
                }
            }
            b_2 = (const_3 * b_2).sqrt();
            let a_2 = d_min_1 / (T::one() + b_2 * b_2);
            let gap_2 = half * d_min_2 - a_2;
            if gap_2 > T::zero() && gap_2 > b_2 * a_2 {
                let tmp_s = a_2 * (T::one() - const_2 * a_2 * (b_2 / gap_2) * b_2);
                if tmp_s > s {
                    s = tmp_s;
                }
            } else {
                let tmp_s = a_2 * (T::one() - const_2 * b_2);
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
        if d_min_2 == d_n_2 && half * *z.offset(nn - 5) < *z.offset(nn - 7) {
            ttype = -10;
            s = third * d_min_2;
            if *z.offset(nn - 5) > *z.offset(nn - 7) {
                return (s, ttype, g);
            }
            let mut b_1 = *z.offset(nn - 5) / *z.offset(nn - 7);
            let mut b_2 = b_1;
            if b_2 != T::zero() {
                let mut i4 = 4 * n0 as isize - 9 + pong;
                while i4 >= 4 * i0 as isize - 1 + pong {
                    if *z.offset(i4) > *z.offset(i4 - 2) {
                        return (s, ttype, g);
                    }
                    b_1 *= *z.offset(i4) / *z.offset(i4 - 2);
                    b_2 += b_1;
                    if hundred * b_1 < b_2 {
                        break;
                    }
                    i4 -= 4;
                }
            }
            b_2 = (const_3 * b_2).sqrt();
            let a_2 = d_min_2 / (T::one() + b_2 * b_2);
            let gap_2 = *z.offset(nn - 7) + *z.offset(nn - 9)
                - (*z.offset(nn - 11)).sqrt() * (*z.offset(nn - 9)).sqrt()
                - a_2;
            if gap_2 > T::zero() && gap_2 > b_2 * a_2 {
                let tmp_s = a_2 * (T::one() - const_2 * a_2 * (b_2 / gap_2) * b_2);
                if tmp_s > s {
                    s = tmp_s;
                }
            } else {
                let tmp_s = a_2 * (T::one() - const_2 * b_2);
                if tmp_s > s {
                    s = tmp_s;
                }
            }
        } else {
            s = quarter * d_min_2;
            ttype = -11;
        }
    } else if n0_in > n0 + 2 {
        s = T::zero();
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
#[allow(clippy::cast_possible_wrap, clippy::too_many_lines)]
pub(crate) unsafe fn lasq5<T: Real>(
    i0: usize,
    n0: usize,
    z: *mut T,
    pp: PingPong,
    mut tau: T,
    sigma: T,
    eps: T,
) -> (T, T, T, T, T, T) {
    assert!(n0 - i0 <= 1);
    let pong = if let PingPong::Pong = pp { 1 } else { 0 };
    let d_thresh = eps * (sigma + tau);
    if tau < d_thresh / (T::one() + T::one()) {
        tau = T::zero();
    }

    let mut j4 = 4 * i0 as isize + pong - 3;
    let mut e_min = *z.offset(j4 + 4);
    let mut d = *z.offset(j4) - tau;
    let mut d_min = d;
    let d_min_2;
    let d_n;
    let d_nm1;
    let d_nm2;
    #[allow(clippy::collapsible_else_if)]
    if tau == T::zero() {
        if let PingPong::Ping = pp {
            for j4 in (4 * i0 as isize..=4 * (n0 as isize - 3)).step_by(4) {
                *z.offset(j4 - 2) = d + *z.offset(j4 - 1);
                let tmp = *z.offset(j4 + 1) / *z.offset(j4 - 2);
                d = d * tmp - tau;
                if d < d_thresh {
                    d = T::zero();
                }
                if d < d_min {
                    d_min = d;
                }
                *z.offset(j4) = *z.offset(j4 - 1) * tmp;
                let e = *z.offset(j4);
                if e < e_min {
                    e_min = e;
                }
            }
        } else {
            for j4 in (4 * i0 as isize..=4 * (n0 as isize - 3)).step_by(4) {
                *z.offset(j4 - 3) = d + *z.offset(j4);
                let tmp = *z.offset(j4 + 2) / *z.offset(j4 - 3);
                d = d * tmp - tau;
                if d < d_thresh {
                    d = T::zero();
                }
                if d < d_min {
                    d_min = d;
                }
                *z.offset(j4 - 1) = *z.offset(j4) * tmp;
                let e = *z.offset(j4 - 1);
                if e < e_min {
                    e_min = e;
                }
            }
        }
    } else {
        if let PingPong::Ping = pp {
            for j4 in (4 * i0 as isize..=4 * (n0 as isize - 3)).step_by(4) {
                *z.offset(j4 - 2) = d + *z.offset(j4 - 1);
                let tmp = *z.offset(j4 + 1) / *z.offset(j4 - 2);
                d = d * tmp - tau;
                if d < d_min {
                    d_min = d;
                }
                *z.offset(j4) = *z.offset(j4 - 1) * tmp;
                let e = *z.offset(j4);
                if e < e_min {
                    e_min = e;
                }
            }
        } else {
            for j4 in (4 * i0 as isize..=4 * (n0 as isize - 3)).step_by(4) {
                *z.offset(j4 - 3) = d + *z.offset(j4);
                let tmp = *z.offset(j4 + 2) / *z.offset(j4 - 3);
                d = d * tmp - tau;
                if d < d_min {
                    d_min = d;
                }
                *z.offset(j4 - 1) = *z.offset(j4) * tmp;
                let e = *z.offset(j4 - 1);
                if e < e_min {
                    e_min = e;
                }
            }
        }
    }

    d_nm2 = d;
    d_min_2 = d_min;
    j4 = 4 * (n0 as isize - 2) - pong;
    let mut j4_p2 = j4 + 2 * pong - 1;
    *z.offset(j4 - 2) = d_nm2 + *z.offset(j4_p2);
    *z.offset(j4) = *z.offset(j4_p2 + 2) * (*z.offset(j4_p2) / *z.offset(j4 - 2));
    d_nm1 = *z.offset(j4_p2 + 2) * (d_nm2 / *z.offset(j4 - 2)) - tau;
    if d_nm1 < d_min {
        d_min = d_nm1;
    }

    let d_min_1 = d_min;
    j4 += 4;
    j4_p2 = j4 + 2 * pong - 1;
    *z.offset(j4 - 2) = d_nm1 + *z.offset(j4_p2);
    *z.offset(j4) = *z.offset(j4_p2 + 2) * (*z.offset(j4_p2) / *z.offset(j4 - 2));
    d_n = *z.offset(j4_p2 + 2) * (d_nm1 / *z.offset(j4 - 2)) - tau;
    if d_n < d_min {
        d_min = d_n;
    }

    *z.offset(j4 + 2) = d_n;
    if let PingPong::Ping = pp {
        *z.offset(4 * n0 as isize) = e_min;
    } else {
        *z.offset(4 * n0 as isize - 1) = e_min;
    }
    (d_min, d_min_1, d_min_2, d_n, d_nm1, d_nm2)
}

/// Computes one dqd transform in ping-pong form.
///
/// # Panics
///
/// Panics if `n0 - i0 <= 1`.
#[allow(dead_code)]
#[allow(clippy::cast_possible_wrap, clippy::too_many_lines)]
pub(crate) unsafe fn lasq6<T: Real>(
    i0: usize,
    n0: usize,
    z: *mut T,
    pp: PingPong,
) -> (T, T, T, T, T, T) {
    assert!(n0 - i0 <= 1);

    let mut j4 = if let PingPong::Ping = pp {
        4 * i0 as isize - 3
    } else {
        4 * i0 as isize - 2
    };
    let mut e_min = *z.offset(j4 + 4);
    let mut d = *z.offset(j4);
    let mut d_min = d;

    if let PingPong::Ping = pp {
        for j4 in (4 * i0 as isize..=4 * (n0 as isize - 3)).step_by(4) {
            *z.offset(j4 - 2) = d + *z.offset(j4 - 1);
            if *z.offset(j4 - 2) == T::zero() {
                *z.offset(j4) = T::zero();
                d = *z.offset(j4 + 1);
                d_min = d;
                e_min = T::zero();
            } else if T::sfmin() * *z.offset(j4 + 1) < *z.offset(j4 - 2)
                && T::sfmin() * *z.offset(j4 - 2) < *z.offset(j4 + 1)
            {
                let tmp = *z.offset(j4 + 1) / *z.offset(j4 - 2);
                *z.offset(j4) = *z.offset(j4 - 1) * tmp;
                d *= tmp;
            } else {
                *z.offset(j4) = *z.offset(j4 + 1) * (*z.offset(j4 - 1) / *z.offset(j4 - 2));
                d = *z.offset(j4 + 1) * (d / *z.offset(j4 - 2));
            }
            if d < d_min {
                d_min = d;
            }
            let e = *z.offset(j4);
            if e < e_min {
                e_min = e;
            }
        }
    } else {
        for j4 in (4 * i0 as isize..=4 * (n0 as isize - 3)).step_by(4) {
            *z.offset(j4 - 3) = d + *z.offset(j4);
            if *z.offset(j4 - 3) == T::zero() {
                *z.offset(j4 - 1) = T::zero();
                d = *z.offset(j4 + 2);
                d_min = d;
                e_min = T::zero();
            } else if T::sfmin() * *z.offset(j4 + 2) < *z.offset(j4 - 3)
                && T::sfmin() * *z.offset(j4 - 3) < *z.offset(j4 + 2)
            {
                let tmp = *z.offset(j4 + 2) / *z.offset(j4 - 3);
                *z.offset(j4 - 1) = *z.offset(j4) * tmp;
                d *= tmp;
            } else {
                *z.offset(j4) = *z.offset(j4 + 2) * (*z.offset(j4) / *z.offset(j4 - 3));
                d = *z.offset(j4 + 2) * (d / *z.offset(j4 - 3));
            }
            if d < d_min {
                d_min = d;
            }
            let e = *z.offset(j4 - 1);
            if e < e_min {
                e_min = e;
            }
        }
    }

    let d_nm2 = d;
    let d_min_2 = d_min;
    j4 = if let PingPong::Ping = pp {
        4 * (n0 as isize - 2)
    } else {
        4 * (n0 as isize - 2) - 1
    };
    let mut j4_p2 = if let PingPong::Ping = pp {
        j4 - 1
    } else {
        j4 + 1
    };
    *z.offset(j4 - 2) = d_nm2 + *z.offset(j4_p2);
    let d_nm1;
    if *z.offset(j4 - 2) == T::zero() {
        *z.offset(j4) = T::zero();
        d_nm1 = *z.offset(j4_p2 + 2);
        d_min = d_nm1;
        e_min = T::zero();
    } else if T::sfmin() * *z.offset(j4_p2 + 2) < *z.offset(j4 - 2)
        && T::sfmin() * *z.offset(j4 - 2) < *z.offset(j4_p2 + 2)
    {
        let tmp = *z.offset(j4_p2 + 2) / *z.offset(j4 - 2);
        *z.offset(j4) = *z.offset(j4_p2) * tmp;
        d_nm1 = d_nm2 * tmp;
    } else {
        *z.offset(j4) = *z.offset(j4_p2 + 2) * (*z.offset(j4_p2) / *z.offset(j4 - 2));
        d_nm1 = *z.offset(j4_p2 + 2) * (d_nm2 / *z.offset(j4 - 2));
    }

    let d_min_1 = d_min;
    j4 += 4;
    j4_p2 = if let PingPong::Ping = pp {
        j4 - 1
    } else {
        j4 + 1
    };
    *z.offset(j4 - 2) = d_nm1 + *z.offset(j4_p2);
    let d_n;
    if *z.offset(j4 - 2) == T::zero() {
        *z.offset(j4) = T::zero();
        d_n = *z.offset(j4_p2 + 2);
        d_min = d_n;
        e_min = T::zero();
    } else if T::sfmin() * *z.offset(j4_p2 + 2) < *z.offset(j4 - 2)
        && T::sfmin() * *z.offset(j4 - 2) < *z.offset(j4_p2 + 2)
    {
        let tmp = *z.offset(j4_p2 + 2) / *z.offset(j4 - 2);
        *z.offset(j4) = *z.offset(j4_p2) / tmp;
        d_n = d_nm1 * tmp;
    } else {
        *z.offset(j4) = *z.offset(j4_p2 + 2) * (*z.offset(j4_p2) / *z.offset(j4 - 2));
        d_n = *z.offset(j4_p2 + 2) / (d_nm1 / *z.offset(j4 - 2));
    }
    if d_n < d_min {
        d_min = d_n;
    }

    *z.offset(j4 + 2) = d_n;
    if let PingPong::Ping = pp {
        *z.offset(4 * n0 as isize) = e_min;
    } else {
        *z.offset(4 * n0 as isize - 1) = e_min;
    }
    (d_min, d_min_1, d_min_2, d_n, d_nm1, d_nm2)
}
