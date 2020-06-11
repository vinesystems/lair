//! Solving equations.

use crate::{lapack, InvalidInput};
use ndarray::{Array1, ArrayBase, Data, Ix1, Ix2, NdFloat};

/// Solves a system of linear scalar equations.
///
/// # Errors
///
/// * [`InvalidInput::Shape`] if `a` is not a square matrix, or `a`'s number of
///   rows  and `b`'s length does not match.
/// * [`InvalidInput::Value`] if `a` is a singular matrix.
///
/// [`InvalidInput::Shape`]: ../enum.InvalidInput.html#variant.Shape
/// [`InvalidInput::Value`]: ../enum.InvalidInput.html#variant.Value
///
/// # Examples
///
/// ```
/// use lair::solving::solve;
///
/// // Solve the system of equations:
/// //   3 * x0 + x1 = 9
/// //   x0 + 2 * x1 = 8
/// let a = ndarray::arr2(&[[3., 1.], [1., 2.]]);
/// let b = ndarray::arr1(&[9., 8.]);
/// let x = solve(&a, &b).expect("valid input");
/// assert_eq!(x, ndarray::aview1(&[2., 3.]));
/// ```
pub fn solve<A, SA, SB>(
    a: &ArrayBase<SA, Ix2>,
    b: &ArrayBase<SB, Ix1>,
) -> Result<Array1<A>, InvalidInput>
where
    A: NdFloat,
    SA: Data<Elem = A>,
    SB: Data<Elem = A>,
{
    if !a.is_square() {
        return Err(InvalidInput::Shape(
            "input matrix is not square".to_string(),
        ));
    }
    if b.len() != a.nrows() {
        return Err(InvalidInput::Shape(format!(
            "The number of elements in `b`, {}, must be the same as the number of rows in `a`, {}",
            b.len(),
            a.nrows()
        )));
    }
    let mut a = a.to_owned();
    let p = match lapack::getrf(&mut a) {
        Ok(p) => p,
        Err(_) => return Err(InvalidInput::Value("`a` is a singular matrix".to_string())),
    };
    Ok(lapack::getrs(&a, &p, b))
}
