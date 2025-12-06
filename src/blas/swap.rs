use ndarray::{ArrayBase, Axis, DataMut, Ix1};

/// Swaps two vectors.
///
/// For each pair of elements `(zx[i], zy[i])`, exchanges their values.
///
/// This is the Rust equivalent of the BLAS `zswap` routine.
///
/// # Panics
///
/// Panics if `zx` or `zy` has fewer than `n` elements.
#[allow(dead_code)]
#[inline]
pub(crate) fn swap<T, S>(n: usize, zx: &mut ArrayBase<S, Ix1>, zy: &mut ArrayBase<S, Ix1>)
where
    T: Copy,
    S: DataMut<Elem = T>,
{
    assert!(zx.len() >= n && zy.len() >= n);

    let inc_x = zx.stride_of(Axis(0));
    let inc_y = zy.stride_of(Axis(0));
    unsafe { inner(n, zx.as_mut_ptr(), inc_x, zy.as_mut_ptr(), inc_y) }
}

/// Swaps two vectors.
///
/// # Safety
///
/// * `zx` is the beginning address of an array of at least `n` elements with
///   stride `inc_x`.
/// * `zy` is the beginning address of an array of at least `n` elements with
///   stride `inc_y`.
/// * The `n` elements of `zx` and `zy` must have been initialized.
/// * `(n - 1) * inc_x` and `(n - 1) * inc_y` are between `isize::MIN` and
///   `isize::MAX`, inclusive.
#[allow(clippy::cast_possible_wrap)]
unsafe fn inner<T: Copy>(n: usize, zx: *mut T, inc_x: isize, zy: *mut T, inc_y: isize) {
    if n == 0 {
        return;
    }

    if inc_x == 1 && inc_y == 1 {
        // Code for both increments equal to 1
        for i in 0..n {
            let ztemp = *zx.add(i);
            *zx.add(i) = *zy.add(i);
            *zy.add(i) = ztemp;
        }
    } else {
        // Code for unequal increments or equal increments not equal to 1
        let mut ix: isize = 0;
        let mut iy: isize = 0;
        if inc_x < 0 {
            ix = (-(n as isize) + 1) * inc_x;
        }
        if inc_y < 0 {
            iy = (-(n as isize) + 1) * inc_y;
        }
        for _ in 0..n {
            let ztemp = *zx.offset(ix);
            *zx.offset(ix) = *zy.offset(iy);
            *zy.offset(iy) = ztemp;
            ix += inc_x;
            iy += inc_y;
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::arr1;
    use num_complex::Complex64;

    use super::swap;

    #[test]
    fn complex_unit_increment() {
        let mut zx = arr1(&[
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0),
        ]);
        let mut zy = arr1(&[
            Complex64::new(7.0, 8.0),
            Complex64::new(9.0, 10.0),
            Complex64::new(11.0, 12.0),
        ]);

        swap::<Complex64, _>(3, &mut zx, &mut zy);

        // After swap, zx should have zy's original values and vice versa
        assert_eq!(zx[0], Complex64::new(7.0, 8.0));
        assert_eq!(zx[1], Complex64::new(9.0, 10.0));
        assert_eq!(zx[2], Complex64::new(11.0, 12.0));
        assert_eq!(zy[0], Complex64::new(1.0, 2.0));
        assert_eq!(zy[1], Complex64::new(3.0, 4.0));
        assert_eq!(zy[2], Complex64::new(5.0, 6.0));
    }

    #[test]
    fn complex_zero_elements() {
        let mut zx = arr1(&[Complex64::new(1.0, 2.0)]);
        let mut zy = arr1(&[Complex64::new(3.0, 4.0)]);

        swap::<Complex64, _>(0, &mut zx, &mut zy);

        // Arrays should be unchanged
        assert_eq!(zx[0], Complex64::new(1.0, 2.0));
        assert_eq!(zy[0], Complex64::new(3.0, 4.0));
    }

    #[test]
    fn real_vectors() {
        let mut x = arr1(&[1.0, 2.0, 3.0]);
        let mut y = arr1(&[4.0, 5.0, 6.0]);

        swap::<f64, _>(3, &mut x, &mut y);

        assert_abs_diff_eq!(x[0], 4.0);
        assert_abs_diff_eq!(x[1], 5.0);
        assert_abs_diff_eq!(x[2], 6.0);
        assert_abs_diff_eq!(y[0], 1.0);
        assert_abs_diff_eq!(y[1], 2.0);
        assert_abs_diff_eq!(y[2], 3.0);
    }

    #[test]
    fn partial_swap() {
        let mut zx = arr1(&[
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0),
        ]);
        let mut zy = arr1(&[
            Complex64::new(7.0, 8.0),
            Complex64::new(9.0, 10.0),
            Complex64::new(11.0, 12.0),
        ]);

        // Only swap first 2 elements
        swap::<Complex64, _>(2, &mut zx, &mut zy);

        assert_eq!(zx[0], Complex64::new(7.0, 8.0));
        assert_eq!(zx[1], Complex64::new(9.0, 10.0));
        assert_eq!(zx[2], Complex64::new(5.0, 6.0)); // Unchanged
        assert_eq!(zy[0], Complex64::new(1.0, 2.0));
        assert_eq!(zy[1], Complex64::new(3.0, 4.0));
        assert_eq!(zy[2], Complex64::new(11.0, 12.0)); // Unchanged
    }

    #[test]
    fn single_element() {
        let mut zx = arr1(&[Complex64::new(1.0, 2.0)]);
        let mut zy = arr1(&[Complex64::new(3.0, 4.0)]);

        swap::<Complex64, _>(1, &mut zx, &mut zy);

        assert_eq!(zx[0], Complex64::new(3.0, 4.0));
        assert_eq!(zy[0], Complex64::new(1.0, 2.0));
    }

    #[test]
    fn integer_vectors() {
        let mut x = arr1(&[1, 2, 3, 4]);
        let mut y = arr1(&[5, 6, 7, 8]);

        swap::<i32, _>(4, &mut x, &mut y);

        assert_eq!(x[0], 5);
        assert_eq!(x[1], 6);
        assert_eq!(x[2], 7);
        assert_eq!(x[3], 8);
        assert_eq!(y[0], 1);
        assert_eq!(y[1], 2);
        assert_eq!(y[2], 3);
        assert_eq!(y[3], 4);
    }
}
