/// Copies a vector to another vector.
///
/// # Safety
///
/// * `x` is the beginning address of an array of at least `n` elements with
///   stride `inc_x`.
/// * `y` is the beginning address of an array of at least `n` elements with
///   stride `inc_y`.
/// * The `n` elements of `x` must have been initialized.
#[allow(dead_code)]
pub(crate) unsafe fn copy<T: Copy>(n: usize, x: *const T, inc_x: isize, y: *mut T, inc_y: isize) {
    let x_ptr = x;
    let y_ptr = y;
    for _ in 0..n {
        *y_ptr = *x_ptr;
        x_ptr.offset(inc_x);
        y_ptr.offset(inc_y);
    }
}
