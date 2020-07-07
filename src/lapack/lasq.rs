use crate::Real;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum PingPong {
    Ping,
    Pong,
}

/// Computes one dqd transform in ping-pong form.
#[allow(dead_code)]
#[allow(clippy::cast_possible_wrap, clippy::too_many_lines)]
pub(crate) unsafe fn lasq6<T: Real>(
    i0: usize,
    n0: usize,
    z: *mut T,
    pp: PingPong,
) -> (T, T, T, T, T, T) {
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
