use num_traits::Float;

/// Multiplies a slice by a real scalar `c_to / c_from`, using iterative
/// scaling to avoid overflow/underflow.
///
/// This is the "general" (type 'G') case of LAPACK's DLASCL.
#[allow(dead_code)]
pub(crate) fn general<A: Float>(c_from: A, c_to: A, a: &mut [A]) {
    let small_num = A::min_positive_value();
    let large_num = A::one() / small_num;

    let mut cfromc = c_from;
    let mut ctoc = c_to;

    loop {
        let cfrom_small = cfromc * small_num;
        let (mul, done) = if cfrom_small == cfromc {
            // cfromc is inf: multiply by a correctly signed zero for finite ctoc,
            // or a NaN if ctoc is infinite.
            (ctoc / cfromc, true)
        } else {
            let cto_big = ctoc / large_num;
            if cto_big == ctoc {
                // ctoc is either 0 or inf. In both cases, ctoc itself serves as
                // the correct multiplication factor.
                (ctoc, true)
            } else if cfrom_small.abs() > ctoc.abs() && ctoc != A::zero() {
                // cfromc is large relative to ctoc: scale down iteratively
                cfromc = cfrom_small;
                (small_num, false)
            } else if cto_big.abs() > cfromc.abs() {
                // ctoc is large relative to cfromc: scale up iteratively
                ctoc = cto_big;
                (large_num, false)
            } else {
                // Normal case: direct scaling is safe
                (ctoc / cfromc, true)
            }
        };

        for v in a.iter_mut() {
            *v = *v * mul;
        }

        if done {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    #[test]
    fn general_simple() {
        let mut a = [3.0f64, 2.0, 4.0, 3.0];
        super::general(2.0, 4.0, &mut a);
        assert_eq!(a, [6.0, 4.0, 8.0, 6.0]);
    }

    #[test]
    fn general_scale_down() {
        // Scale down by a factor of 10
        let mut a = [30.0f64, 20.0, 40.0];
        super::general(10.0, 1.0, &mut a);
        assert_eq!(a, [3.0, 2.0, 4.0]);
    }

    #[test]
    fn general_extreme_scale_up() {
        // Scale by a very large factor - should use iterative scaling
        let mut a = [1.0f64, 2.0];
        let scale = (f64::EPSILON / f64::MIN_POSITIVE).sqrt();
        super::general(1.0, scale, &mut a);
        // Result should be finite, not overflow
        assert!(a[0].is_finite());
        assert!(a[1].is_finite());
        // And approximately correct
        assert_abs_diff_eq!(a[0], scale, epsilon = scale * 1e-10);
        assert_abs_diff_eq!(a[1], 2.0 * scale, epsilon = scale * 1e-10);
    }

    #[test]
    fn general_extreme_scale_down() {
        // Scale by a very small factor
        let mut a = [1.0f64, 2.0];
        let scale = (f64::EPSILON / f64::MIN_POSITIVE).sqrt();
        super::general(scale, 1.0, &mut a);
        // Result should be finite
        assert!(a[0].is_finite());
        assert!(a[1].is_finite());
        // And approximately correct
        assert_abs_diff_eq!(a[0], 1.0 / scale, epsilon = 1.0 / scale * 1e-10);
    }

    #[test]
    fn general_empty() {
        let mut a: [f64; 0] = [];
        super::general(2.0, 4.0, &mut a);
        // Should not panic
    }

    #[test]
    fn general_f32() {
        let mut a = [3.0f32, 2.0];
        super::general(2.0f32, 6.0f32, &mut a);
        assert_eq!(a, [9.0, 6.0]);
    }
}
