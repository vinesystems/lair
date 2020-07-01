use crate::Real;

/// Computes singular values of a 2x2 triangular matrix.
#[allow(dead_code)]
fn las2<T: Real>(f: T, g: T, h: T) -> (T, T) {
    let f_abs = f.abs();
    let g_abs = g.abs();
    let h_abs = h.abs();
    let (f_h_min, f_h_max) = if f_abs < h_abs {
        (f_abs, h_abs)
    } else {
        (h_abs, f_abs)
    };
    if f_h_min == T::zero() {
        if f_h_max == T::zero() {
            (f_h_min, g_abs)
        } else {
            let f_h_g_max = if f_h_max > g_abs { f_h_max } else { g_abs };
            let smaller = if f_h_max < g_abs { f_h_max } else { g_abs };
            let ratio = smaller / f_h_g_max;
            (f_h_min, f_h_g_max * (T::one() + ratio * ratio).sqrt())
        }
    } else if g_abs < f_h_max {
        let a_s = T::one() + f_h_min / f_h_max;
        let a_t = (f_h_max - f_h_min) / f_h_max;
        let ratio = g_abs / f_h_max;
        let a_u = ratio * ratio;
        let c = (T::one() + T::one()) / ((a_s * a_s + a_u).sqrt() + (a_t * a_t + a_u).sqrt());
        (f_h_min * c, f_h_max / c)
    } else {
        let a_u = f_h_max / g_abs;
        if a_u == T::zero() {
            (f_h_min * f_h_max / g_abs, g_abs)
        } else {
            let a_s = T::one() + f_h_min / f_h_max;
            let a_t = (f_h_max - f_h_min) / f_h_max;
            let s_u = a_s * a_u;
            let t_u = a_t * a_u;
            let c = T::one() / ((T::one() + s_u * s_u).sqrt() + (T::one() + t_u * t_u).sqrt());
            let half_min = f_h_min * c * a_u;
            (half_min + half_min, g_abs / (c + c))
        }
    }
}
