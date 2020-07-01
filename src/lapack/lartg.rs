use crate::Real;

/// Generates a plane rotation with real cosine and real sine.
#[allow(dead_code)]
pub(crate) fn lartg<T: Real>(f: T, g: T) -> (T, T, T) {
    if g == T::zero() {
        return (T::one(), T::zero(), f);
    }
    if f == T::zero() {
        return (T::zero(), T::one(), g);
    }

    let two = T::one() + T::one();
    let safe_min = two.powf(((T::sfmin() / T::eps()).log2() / (T::one() + T::one())).trunc());
    let safe_max = safe_min.recip();

    let larger = if f.abs() > g.abs() { f.abs() } else { g.abs() };
    let (cos, sin, r) = if larger >= safe_max {
        let mut f = f;
        let mut g = g;
        let mut count = 0;
        loop {
            count += 1;
            f *= safe_min;
            g *= safe_min;
            let scale = if f.abs() > g.abs() { f.abs() } else { g.abs() };
            if scale < safe_max {
                break;
            }
        }
        let mut r = (f * f + g * g).sqrt();
        let cos = f / r;
        let sin = g / r;
        for _ in 0..count {
            r *= safe_max;
        }
        (cos, sin, r)
    } else if larger <= safe_min {
        let mut f = f;
        let mut g = g;
        let mut count = 0;
        loop {
            count += 1;
            f *= safe_max;
            g *= safe_max;
            let scale = if f.abs() > g.abs() { f.abs() } else { g.abs() };
            if scale > safe_min {
                break;
            }
        }
        let mut r = (f * f + g * g).sqrt();
        let cos = f / r;
        let sin = g / r;
        for _ in 0..count {
            r *= safe_min;
        }
        (cos, sin, r)
    } else {
        let r = (f * f + g * g).sqrt();
        (f / r, g / r, r)
    };
    if f.abs() > g.abs() && cos < T::zero() {
        (-cos, -sin, -r)
    } else {
        (cos, sin, r)
    }
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;

    #[test]
    fn lartg_nan() {
        let (cos, sin, r) = super::lartg(f32::NAN, 1.);
        assert!(cos.is_nan());
        assert!(sin.is_nan());
        assert!(r.is_nan());

        let (cos, sin, r) = super::lartg(2., f64::NAN);
        assert!(cos.is_nan());
        assert!(sin.is_nan());
        assert!(r.is_nan());

        let (cos, sin, r) = super::lartg(f64::NAN, 0.);
        assert_eq!(cos, 1.);
        assert_eq!(sin, 0.);
        assert!(r.is_nan());

        let (cos, sin, r) = super::lartg(0., f32::NAN);
        assert_eq!(cos, 0.);
        assert_eq!(sin, 1.);
        assert!(r.is_nan());
    }

    #[test]
    fn lartg() {
        let (cos, sin, r) = super::lartg(0., 2.);
        assert_eq!(cos, 0.);
        assert_eq!(sin, 1.);
        assert_eq!(r, 2.);

        let (cos, sin, r) = super::lartg(2., 0.);
        assert_eq!(cos, 1.);
        assert_eq!(sin, 0.);
        assert_eq!(r, 2.);

        let (cos, sin, r) = super::lartg(2., 3.);
        assert_abs_diff_eq!(cos, 0.5547001962252291, epsilon = 1e-8);
        assert_abs_diff_eq!(sin, 0.8320502943378437, epsilon = 1e-8);
        assert_abs_diff_eq!(r, 3.605551275463989, epsilon = 1e-8);
    }
}
