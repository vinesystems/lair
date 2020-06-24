//! Matrix functions and special matrices.

use crate::{InvalidInput, Scalar};
use ndarray::{Array2, ArrayBase};

/// Constructs a circulant matrix.
///
/// # Examples
///
/// ```
/// use lair::matrix::circulant;
///
/// let a = vec![1., 2., 3.];
/// let c = circulant(&a);
/// assert_eq!(c, ndarray::aview2(&[[1., 3., 2.], [2., 1., 3.], [3., 2., 1.]]));
/// ```
pub fn circulant<A>(a: &[A]) -> Array2<A>
where
    A: Copy,
{
    unsafe {
        let mut x: Array2<A> = ArrayBase::uninitialized((a.len(), a.len()));
        for (i, a_elem) in a.iter().enumerate() {
            for j in 0..a.len() {
                x[[(i + j) % a.len(), j]] = *a_elem;
            }
        }
        x
    }
}

/// Constructs a companion matrix.
///
/// # Errors
///
/// * [`InvalidInput::Shape`] if `a` contains less than two coefficients.
/// * [`InvalidInput::Value`] if `a[0]` is zero.
///
/// [`InvalidInput::Shape`]: ../enum.InvalidInput.html#variant.Shape
/// [`InvalidInput::Value`]: ../enum.InvalidInput.html#variant.Value
///
/// # Examples
///
/// ```
/// use lair::matrix::companion;
///
/// let a = vec![1., -10., 31., -30.];
/// let c = companion(&a).expect("valid input");
/// assert_eq!(c, ndarray::aview2(&[[10., -31., 30.], [1., 0., 0.], [0., 1., 0.]]));
/// ```
pub fn companion<A>(a: &[A]) -> Result<Array2<A>, InvalidInput>
where
    A: Scalar,
{
    if a.len() < 2 {
        return Err(InvalidInput::Shape(format!(
            "input polynomial has {} coefficient; expected at least two",
            a.len()
        )));
    }
    if a[0] == A::zero() {
        return Err(InvalidInput::Value(format!(
            "invalid first coefficient {}, expected a non-zero value",
            a[0]
        )));
    }

    let mut matrix = Array2::<A>::zeros((a.len() - 1, a.len() - 1));
    for (mv, av) in matrix.row_mut(0).into_iter().zip(a.iter().skip(1)) {
        *mv = -*av / a[0];
    }
    for i in 1..a.len() - 1 {
        matrix[[i, i - 1]] = A::one();
    }
    Ok(matrix)
}

#[cfg(test)]
mod test {
    use ndarray::aview2;
    use num_complex::{Complex32, Complex64};

    #[test]
    fn circulant() {
        let a = Vec::<f32>::new();
        let c = super::circulant(&a);
        assert_eq!(c.shape(), [0, 0]);

        let a = vec![Complex32::new(1., 2.)];
        let c = super::circulant(&a);
        assert_eq!(c, aview2(&[[Complex32::new(1., 2.)]]));

        let a = vec![1., 2., 4.];
        let c = super::circulant(&a);
        assert_eq!(c, aview2(&[[1., 4., 2.], [2., 1., 4.], [4., 2., 1.]]));
    }

    #[test]
    fn companion() {
        let a = Vec::<f32>::new();
        assert!(super::companion(&a).is_err());

        let a = vec![Complex64::new(1., 2.)];
        assert!(super::companion(&a).is_err());

        let a = vec![
            Complex32::new(0., 0.),
            Complex32::new(1., 1.),
            Complex32::new(2., 2.),
        ];
        assert!(super::companion(&a).is_err());

        let a = vec![Complex32::new(1., 2.), Complex32::new(3., 4.)];
        let c = super::companion(&a).expect("valid input");
        assert_eq!(c, aview2(&[[Complex32::new(-2.2, 0.4)]]));

        let a = vec![2., -4., 8., -10.];
        let c = super::companion(&a).expect("valid input");
        assert_eq!(
            c,
            aview2(&[[2.0, -4.0, 5.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        );
    }
}
