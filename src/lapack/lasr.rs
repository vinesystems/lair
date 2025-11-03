use ndarray::{ArrayBase, DataMut, Ix2};
use num_traits::{One, Zero};

use crate::Scalar;

/// Specifies whether the plane rotation matrix P is applied from the left or right.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Side {
    /// Left: compute A := P*A
    Left,
    /// Right: compute A := A*P^T
    Right,
}

/// Specifies the plane for which P(k) is a plane rotation matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Pivot {
    /// Variable pivot: the plane (k, k+1)
    Variable,
    /// Top pivot: the plane (1, k+1)
    Top,
    /// Bottom pivot: the plane (k, z)
    Bottom,
}

/// Specifies whether P is a forward or backward sequence of plane rotations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Direction {
    /// Forward: P = P(z-1) * ... * P(2) * P(1)
    Forward,
    /// Backward: P = P(1) * P(2) * ... * P(z-1)
    Backward,
}

/// Applies a sequence of real plane rotations to a matrix.
///
/// LASR applies a sequence of real plane rotations to a matrix A,
/// from either the left or the right.
///
/// When `side = Side::Left`, the transformation takes the form A := P*A
/// and when `side = Side::Right`, the transformation takes the form A := A*P^T
///
/// where P is an orthogonal matrix consisting of a sequence of z plane rotations,
/// with z = M when SIDE = Left and z = N when SIDE = Right, and P^T is the
/// transpose of P.
///
/// The 2-by-2 plane rotation part of the matrix P(k), R(k), has the form:
///
/// ```text
/// R(k) = (  c(k)  s(k) )
///        ( -s(k)  c(k) )
/// ```
///
/// # Arguments
///
/// * `side` - Specifies whether the plane rotation matrix P is applied from the left or right
/// * `pivot` - Specifies the plane for which P(k) is a plane rotation matrix
/// * `direct` - Specifies whether P is a forward or backward sequence of plane rotations
/// * `c` - Array of cosines for the plane rotations. Length should be M-1 if side = Left, N-1 if side = Right
/// * `s` - Array of sines for the plane rotations. Length should be M-1 if side = Left, N-1 if side = Right
/// * `a` - The M-by-N matrix A. On exit, A is overwritten by P*A if side = Left or by A*P^T if side = Right
///
/// # Panics
///
/// Panics if:
/// * The length of `c` is not M-1 when side = Left or N-1 when side = Right
/// * The length of `s` is not M-1 when side = Left or N-1 when side = Right
/// * The matrix dimensions are incompatible
#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_lines)]
#[allow(dead_code)]
pub fn lasr<A, S>(
    side: Side,
    pivot: Pivot,
    direct: Direction,
    c: &[A::Real],
    s: &[A::Real],
    a: &mut ArrayBase<S, Ix2>,
) where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    let (m, n) = a.dim();

    // Quick return if possible
    if m == 0 || n == 0 {
        return;
    }

    match side {
        Side::Left => {
            // Form P * A
            assert_eq!(c.len(), m.saturating_sub(1), "c must have length M-1");
            assert_eq!(s.len(), m.saturating_sub(1), "s must have length M-1");

            match pivot {
                Pivot::Variable => match direct {
                    Direction::Forward => {
                        for j in 0..m - 1 {
                            let ctemp = c[j];
                            let stemp = s[j];
                            if ctemp != A::Real::one() || stemp != A::Real::zero() {
                                for i in 0..n {
                                    let temp = a[(j + 1, i)];
                                    a[(j + 1, i)] = ctemp.into() * temp - stemp.into() * a[(j, i)];
                                    a[(j, i)] = stemp.into() * temp + ctemp.into() * a[(j, i)];
                                }
                            }
                        }
                    }
                    Direction::Backward => {
                        for j in (0..m - 1).rev() {
                            let ctemp = c[j];
                            let stemp = s[j];
                            if ctemp != A::Real::one() || stemp != A::Real::zero() {
                                for i in 0..n {
                                    let temp = a[(j + 1, i)];
                                    a[(j + 1, i)] = ctemp.into() * temp - stemp.into() * a[(j, i)];
                                    a[(j, i)] = stemp.into() * temp + ctemp.into() * a[(j, i)];
                                }
                            }
                        }
                    }
                },
                Pivot::Top => match direct {
                    Direction::Forward => {
                        for j in 1..m {
                            let ctemp = c[j - 1];
                            let stemp = s[j - 1];
                            if ctemp != A::Real::one() || stemp != A::Real::zero() {
                                for i in 0..n {
                                    let temp = a[(j, i)];
                                    a[(j, i)] = ctemp.into() * temp - stemp.into() * a[(0, i)];
                                    a[(0, i)] = stemp.into() * temp + ctemp.into() * a[(0, i)];
                                }
                            }
                        }
                    }
                    Direction::Backward => {
                        for j in (1..m).rev() {
                            let ctemp = c[j - 1];
                            let stemp = s[j - 1];
                            if ctemp != A::Real::one() || stemp != A::Real::zero() {
                                for i in 0..n {
                                    let temp = a[(j, i)];
                                    a[(j, i)] = ctemp.into() * temp - stemp.into() * a[(0, i)];
                                    a[(0, i)] = stemp.into() * temp + ctemp.into() * a[(0, i)];
                                }
                            }
                        }
                    }
                },
                Pivot::Bottom => match direct {
                    Direction::Forward => {
                        for j in 0..m - 1 {
                            let ctemp = c[j];
                            let stemp = s[j];
                            if ctemp != A::Real::one() || stemp != A::Real::zero() {
                                for i in 0..n {
                                    let temp = a[(j, i)];
                                    a[(j, i)] = stemp.into() * a[(m - 1, i)] + ctemp.into() * temp;
                                    a[(m - 1, i)] =
                                        ctemp.into() * a[(m - 1, i)] - stemp.into() * temp;
                                }
                            }
                        }
                    }
                    Direction::Backward => {
                        for j in (0..m - 1).rev() {
                            let ctemp = c[j];
                            let stemp = s[j];
                            if ctemp != A::Real::one() || stemp != A::Real::zero() {
                                for i in 0..n {
                                    let temp = a[(j, i)];
                                    a[(j, i)] = stemp.into() * a[(m - 1, i)] + ctemp.into() * temp;
                                    a[(m - 1, i)] =
                                        ctemp.into() * a[(m - 1, i)] - stemp.into() * temp;
                                }
                            }
                        }
                    }
                },
            }
        }
        Side::Right => {
            // Form A * P^T
            assert_eq!(c.len(), n.saturating_sub(1), "c must have length N-1");
            assert_eq!(s.len(), n.saturating_sub(1), "s must have length N-1");

            match pivot {
                Pivot::Variable => match direct {
                    Direction::Forward => {
                        for j in 0..n - 1 {
                            let ctemp = c[j];
                            let stemp = s[j];
                            if ctemp != A::Real::one() || stemp != A::Real::zero() {
                                for i in 0..m {
                                    let temp = a[(i, j + 1)];
                                    a[(i, j + 1)] = ctemp.into() * temp - stemp.into() * a[(i, j)];
                                    a[(i, j)] = stemp.into() * temp + ctemp.into() * a[(i, j)];
                                }
                            }
                        }
                    }
                    Direction::Backward => {
                        for j in (0..n - 1).rev() {
                            let ctemp = c[j];
                            let stemp = s[j];
                            if ctemp != A::Real::one() || stemp != A::Real::zero() {
                                for i in 0..m {
                                    let temp = a[(i, j + 1)];
                                    a[(i, j + 1)] = ctemp.into() * temp - stemp.into() * a[(i, j)];
                                    a[(i, j)] = stemp.into() * temp + ctemp.into() * a[(i, j)];
                                }
                            }
                        }
                    }
                },
                Pivot::Top => match direct {
                    Direction::Forward => {
                        for j in 1..n {
                            let ctemp = c[j - 1];
                            let stemp = s[j - 1];
                            if ctemp != A::Real::one() || stemp != A::Real::zero() {
                                for i in 0..m {
                                    let temp = a[(i, j)];
                                    a[(i, j)] = ctemp.into() * temp - stemp.into() * a[(i, 0)];
                                    a[(i, 0)] = stemp.into() * temp + ctemp.into() * a[(i, 0)];
                                }
                            }
                        }
                    }
                    Direction::Backward => {
                        for j in (1..n).rev() {
                            let ctemp = c[j - 1];
                            let stemp = s[j - 1];
                            if ctemp != A::Real::one() || stemp != A::Real::zero() {
                                for i in 0..m {
                                    let temp = a[(i, j)];
                                    a[(i, j)] = ctemp.into() * temp - stemp.into() * a[(i, 0)];
                                    a[(i, 0)] = stemp.into() * temp + ctemp.into() * a[(i, 0)];
                                }
                            }
                        }
                    }
                },
                Pivot::Bottom => match direct {
                    Direction::Forward => {
                        for j in 0..n - 1 {
                            let ctemp = c[j];
                            let stemp = s[j];
                            if ctemp != A::Real::one() || stemp != A::Real::zero() {
                                for i in 0..m {
                                    let temp = a[(i, j)];
                                    a[(i, j)] = stemp.into() * a[(i, n - 1)] + ctemp.into() * temp;
                                    a[(i, n - 1)] =
                                        ctemp.into() * a[(i, n - 1)] - stemp.into() * temp;
                                }
                            }
                        }
                    }
                    Direction::Backward => {
                        for j in (0..n - 1).rev() {
                            let ctemp = c[j];
                            let stemp = s[j];
                            if ctemp != A::Real::one() || stemp != A::Real::zero() {
                                for i in 0..m {
                                    let temp = a[(i, j)];
                                    a[(i, j)] = stemp.into() * a[(i, n - 1)] + ctemp.into() * temp;
                                    a[(i, n - 1)] =
                                        ctemp.into() * a[(i, n - 1)] - stemp.into() * temp;
                                }
                            }
                        }
                    }
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::arr2;
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn left_variable_forward_real() {
        // Simple 90-degree rotation: c=0, s=1
        // R = ( 0  1)  so row0_new = row1_old, row1_new = -row0_old
        //     (-1  0)
        let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let c = vec![0.0];
        let s = vec![1.0];

        lasr::<f64, _>(
            Side::Left,
            Pivot::Variable,
            Direction::Forward,
            &c,
            &s,
            &mut a,
        );

        // After rotation: row 0 becomes old row 1, row 1 becomes -old row 0
        assert_abs_diff_eq!(a[(0, 0)], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(0, 1)], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(1, 0)], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(1, 1)], -2.0, epsilon = 1e-10);
    }

    #[test]
    fn left_variable_forward_complex() {
        // Simple 90-degree rotation with complex matrix
        let mut a = arr2(&[
            [Complex64::new(1.0, 1.0), Complex64::new(2.0, 2.0)],
            [Complex64::new(3.0, 3.0), Complex64::new(4.0, 4.0)],
        ]);
        let c = vec![0.0];
        let s = vec![1.0];

        lasr::<Complex64, _>(
            Side::Left,
            Pivot::Variable,
            Direction::Forward,
            &c,
            &s,
            &mut a,
        );

        // After rotation: row 0 becomes old row 1, row 1 becomes -old row 0
        assert_abs_diff_eq!(a[(0, 0)].re, 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(0, 0)].im, 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(0, 1)].re, 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(0, 1)].im, 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(1, 0)].re, -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(1, 0)].im, -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(1, 1)].re, -2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(1, 1)].im, -2.0, epsilon = 1e-10);
    }

    #[test]
    fn right_variable_forward_real() {
        // Simple 90-degree rotation applied from the right
        // col0_new = col1_old, col1_new = -col0_old
        let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let c = vec![0.0];
        let s = vec![1.0];

        lasr::<f64, _>(
            Side::Right,
            Pivot::Variable,
            Direction::Forward,
            &c,
            &s,
            &mut a,
        );

        // After rotation: col 0 becomes old col 1, col 1 becomes -old col 0
        assert_abs_diff_eq!(a[(0, 0)], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(1, 0)], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(0, 1)], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(1, 1)], -3.0, epsilon = 1e-10);
    }

    #[test]
    fn left_top_forward_real() {
        // Top pivot: rotations always involve row 0
        // First rotation: rows 0 and 1
        let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let c = vec![0.0, 1.0]; // First rotation c=0, s=1; second is identity
        let s = vec![1.0, 0.0];

        lasr::<f64, _>(Side::Left, Pivot::Top, Direction::Forward, &c, &s, &mut a);

        // After first rotation (rows 0 and 1): row0 = old row1, row1 = -old row0
        // Second rotation is identity (c=1, s=0), so no change
        assert_abs_diff_eq!(a[(0, 0)], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(0, 1)], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(1, 0)], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(1, 1)], -2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(2, 0)], 5.0, epsilon = 1e-10); // Unchanged
        assert_abs_diff_eq!(a[(2, 1)], 6.0, epsilon = 1e-10); // Unchanged
    }

    #[test]
    fn left_bottom_forward_real() {
        // Bottom pivot: rotations always involve the last row (row 2)
        // First rotation: rows 0 and 2
        let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let c = vec![0.0, 1.0]; // First rotation c=0, s=1; second is identity
        let s = vec![1.0, 0.0];

        lasr::<f64, _>(
            Side::Left,
            Pivot::Bottom,
            Direction::Forward,
            &c,
            &s,
            &mut a,
        );

        // After first rotation (rows 0 and 2): row0 = s*row2 + c*row0 = row2, row2 = c*row2 - s*row0 = -row0_old
        assert_abs_diff_eq!(a[(0, 0)], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(0, 1)], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(1, 0)], 3.0, epsilon = 1e-10); // Unchanged
        assert_abs_diff_eq!(a[(1, 1)], 4.0, epsilon = 1e-10); // Unchanged
        assert_abs_diff_eq!(a[(2, 0)], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(2, 1)], -2.0, epsilon = 1e-10);
    }

    #[test]
    fn left_variable_backward_real() {
        // Backward sequence: apply rotations in reverse order
        let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let c = vec![1.0, 0.0]; // First is identity, second rotation c=0, s=1
        let s = vec![0.0, 1.0];

        lasr::<f64, _>(
            Side::Left,
            Pivot::Variable,
            Direction::Backward,
            &c,
            &s,
            &mut a,
        );

        // Backward: apply second rotation (rows 1 and 2) first, then first rotation (identity)
        // Second rotation: row1 = row2, row2 = -row1_old
        assert_abs_diff_eq!(a[(0, 0)], 1.0, epsilon = 1e-10); // Unchanged
        assert_abs_diff_eq!(a[(0, 1)], 2.0, epsilon = 1e-10); // Unchanged
        assert_abs_diff_eq!(a[(1, 0)], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(1, 1)], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(2, 0)], -3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(2, 1)], -4.0, epsilon = 1e-10);
    }

    #[test]
    fn empty_matrix() {
        let mut a = arr2::<f64, _>(&[[]]);
        let c = vec![];
        let s = vec![];

        // Should not panic and should do nothing
        lasr::<f64, _>(
            Side::Left,
            Pivot::Variable,
            Direction::Forward,
            &c,
            &s,
            &mut a,
        );
    }

    #[test]
    fn identity_rotation() {
        let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let c = vec![1.0]; // cos = 1 means no rotation
        let s = vec![0.0]; // sin = 0 means no rotation

        lasr::<f64, _>(
            Side::Left,
            Pivot::Variable,
            Direction::Forward,
            &c,
            &s,
            &mut a,
        );

        // Matrix should be unchanged
        assert_abs_diff_eq!(a[(0, 0)], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(0, 1)], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(1, 0)], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[(1, 1)], 4.0, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "c must have length M-1")]
    fn wrong_c_length_left() {
        let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let c = vec![0.8, 0.6]; // Wrong length: should be M-1 = 1
        let s = vec![0.6];

        lasr::<f64, _>(
            Side::Left,
            Pivot::Variable,
            Direction::Forward,
            &c,
            &s,
            &mut a,
        );
    }

    #[test]
    #[should_panic(expected = "c must have length N-1")]
    fn wrong_c_length_right() {
        let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let c = vec![0.8, 0.6]; // Wrong length: should be N-1 = 1
        let s = vec![0.6];

        lasr::<f64, _>(
            Side::Right,
            Pivot::Variable,
            Direction::Forward,
            &c,
            &s,
            &mut a,
        );
    }
}
