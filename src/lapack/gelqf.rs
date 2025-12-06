use std::cmp;
use std::ops::{Div, MulAssign};

use ndarray::{s, Array1, ArrayBase, DataMut, Ix2};

use crate::{lapack, Scalar};

/// Computes the LQ factorization of a matrix.
///
/// The LQ factorization is `A = L * Q`, where L is lower triangular and Q is
/// orthogonal (or unitary in the complex case).
///
/// For an M-by-N matrix A, the factorization is:
/// - `A = (L 0) * Q` when M <= N
/// - Where Q is an N-by-N orthogonal/unitary matrix
/// - L is an M-by-M lower triangular matrix
///
/// The elementary reflectors that represent Q are stored in the upper
/// triangular part of A (above the diagonal), and the scalar factors tau
/// are returned. The diagonal and lower triangular part of A contain L.
#[allow(dead_code)]
pub fn gelqf<A, S>(a: &mut ArrayBase<S, Ix2>) -> Array1<A>
where
    A: Scalar + Div<<A as Scalar>::Real, Output = A> + MulAssign<<A as Scalar>::Real>,
    S: DataMut<Elem = A>,
{
    let min_dim = cmp::min(a.nrows(), a.ncols());
    let mut tau = Array1::<A>::zeros(min_dim);

    for i in 0..min_dim {
        // Conjugate the row to prepare for reflector generation
        lapack::lacgv(&mut a.row_mut(i).slice_mut(s![i..]));

        // Generate elementary reflector H(i) to annihilate A(i, i+1:n)
        let right = a.ncols();
        let (beta, _, t) = lapack::larfg(a[(i, i)], a.row_mut(i).slice_mut(s![i + 1..right]));
        tau[i] = t;

        if i + 1 < a.nrows() {
            // Apply H(i) to A(i+1:m, i:n) from the right
            a[(i, i)] = A::one();
            let v = a.row(i).slice(s![i..]).to_owned();
            lapack::larf::right(&v, t, &mut a.slice_mut(s![i + 1.., i..]));
            a[(i, i)] = beta.into();
        } else {
            a[(i, i)] = beta.into();
        }

        // Unconjugate the row (except the diagonal element which is real)
        if i + 1 < a.ncols() {
            lapack::lacgv(&mut a.row_mut(i).slice_mut(s![i + 1..]));
        }
    }
    tau
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2};
    use num_complex::Complex64;

    #[test]
    fn square_real_smallest() {
        let a = arr2(&[[2_f64]]);
        let mut lq = a.clone();
        assert_eq!(lq.shape(), &[1, 1]);
        let tau = super::gelqf(&mut lq);
        assert_eq!(tau.shape(), &[1]);
        assert_abs_diff_eq!(lq, arr2(&[[2_f64]]), epsilon = 1e-8);
        assert_abs_diff_eq!(tau, arr1(&[0.]), epsilon = 1e-8);
    }

    #[test]
    fn square_real() {
        let a = arr2(&[
            [1_f64, 0_f64, 0_f64],
            [2_f64, 0_f64, 3_f64],
            [4_f64, 5_f64, 6_f64],
        ]);
        let mut lq = a.clone();
        let tau = super::gelqf(&mut lq);
        assert_eq!(tau.shape(), &[3]);
        assert_abs_diff_eq!(lq[(0, 0)], 1., epsilon = 1e-8);
        assert_abs_diff_eq!(lq[(0, 1)], 0., epsilon = 1e-8);
        assert_abs_diff_eq!(lq[(0, 2)], 0., epsilon = 1e-8);
        assert_abs_diff_eq!(lq[(1, 0)], 2., epsilon = 1e-8);
        assert_abs_diff_eq!(lq[(2, 0)], 4., epsilon = 1e-8);
        assert_abs_diff_eq!(tau[0], 0., epsilon = 1e-8);
    }

    #[test]
    fn square_complex() {
        let a = arr2(&[
            [Complex64::new(1., 1.), Complex64::new(3., 1.)],
            [Complex64::new(2., -1.), Complex64::new(4., -1.)],
        ]);
        let mut lq = a.clone();
        let tau = super::gelqf(&mut lq);
        assert_eq!(tau.shape(), &[2]);
        // LQ factorization: L is lower triangular
        // The result should have the L factor on the diagonal and below,
        // with reflector information above
        assert_abs_diff_eq!(lq[(0, 0)].re, -3.46410162, epsilon = 1e-6);
        assert_abs_diff_eq!(lq[(0, 0)].im, 0., epsilon = 1e-6);
    }

    #[test]
    fn wide() {
        // M < N case - typical for LQ
        let a = arr2(&[
            [1_f64, 2_f64, 3_f64, 1_f64],
            [2_f64, 2_f64, 1_f64, 3_f64],
            [3_f64, 1_f64, 2_f64, 2_f64],
        ]);
        let mut lq = a.clone();
        let tau = super::gelqf(&mut lq);
        assert_eq!(tau.shape(), &[3]);
        // Verify L is computed (diagonal and below)
        assert_abs_diff_eq!(lq[(0, 0)], -3.872983346207417, epsilon = 1e-6);
    }

    #[test]
    fn tall() {
        // M > N case
        let a = arr2(&[
            [1_f64, 2_f64, 3_f64],
            [2_f64, 2_f64, 1_f64],
            [3_f64, 1_f64, 2_f64],
            [2_f64, 3_f64, 3_f64],
        ]);
        let mut lq = a.clone();
        let tau = super::gelqf(&mut lq);
        assert_eq!(tau.shape(), &[3]);
        // The first 3 rows should contain LQ factorization
        assert_abs_diff_eq!(lq[(0, 0)], -3.7416573867739413, epsilon = 1e-6);
    }

    #[test]
    fn verify_lq_real() {
        // Verify that A = L * Q by reconstructing the matrix
        use ndarray::Array2;

        use crate::lapack;

        let a = arr2(&[[1_f64, 2_f64, 3_f64], [4_f64, 5_f64, 6_f64]]);
        let mut lq = a.clone();
        let tau = super::gelqf(&mut lq);

        // For M×N (2×3) matrix:
        // - L is M×M (2×2) lower triangular
        // - Q is M×N (2×3) - first M rows of full orthogonal matrix

        // Extract L (lower triangular part, M×M = 2×2)
        let mut l = Array2::<f64>::zeros((2, 2));
        for i in 0..2 {
            for j in 0..=i {
                l[(i, j)] = lq[(i, j)];
            }
        }

        // Reconstruct Q using unglq (produces M×N matrix)
        let mut q = lq.clone();
        lapack::unglq(&mut q, &tau);

        // Compute L * Q (2×2 * 2×3 = 2×3)
        let reconstructed = l.dot(&q);

        // Compare with original A
        for i in 0..2 {
            for j in 0..3 {
                assert_abs_diff_eq!(reconstructed[(i, j)], a[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn verify_lq_complex() {
        use crate::lapack;

        let a = arr2(&[
            [Complex64::new(1., 2.), Complex64::new(3., 4.)],
            [Complex64::new(5., 6.), Complex64::new(7., 8.)],
        ]);
        let mut lq = a.clone();
        let tau = super::gelqf(&mut lq);

        // Extract L (lower triangular part)
        let mut l = arr2(&[[Complex64::new(0., 0.); 2]; 2]);
        for i in 0..2 {
            for j in 0..=i {
                l[(i, j)] = lq[(i, j)];
            }
        }

        // Reconstruct Q using unglq
        let mut q = lq.clone();
        lapack::unglq(&mut q, &tau);

        // Compute L * Q
        let reconstructed = l.dot(&q);

        // Compare with original A
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(reconstructed[(i, j)].re, a[(i, j)].re, epsilon = 1e-10);
                assert_abs_diff_eq!(reconstructed[(i, j)].im, a[(i, j)].im, epsilon = 1e-10);
            }
        }
    }
}
