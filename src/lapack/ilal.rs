use crate::Scalar;
use ndarray::{ArrayBase, Data, Ix2};

/// Finds the last non-zero column in a matrix.
pub fn ilalc<A, S>(x: &ArrayBase<S, Ix2>) -> Option<usize>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    if x.nrows() == 0 || x.ncols() == 0 {
        return None;
    }

    let last_col = x.ncols() - 1;
    if x[(x.nrows() - 1, last_col)] != A::zero() {
        return Some(last_col);
    }

    for col in (0..x.ncols()).rev() {
        if x.column(col).iter().any(|&elem| elem != A::zero()) {
            return Some(col);
        }
    }
    None
}

/// Finds the last non-zero row in a matrix.
pub fn ilalr<A, S>(x: &ArrayBase<S, Ix2>) -> Option<usize>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    if x.nrows() == 0 || x.ncols() == 0 {
        return None;
    }

    let last_row = x.nrows() - 1;
    if x[(last_row, x.ncols() - 1)] != A::zero() {
        return Some(last_row);
    }

    for row in (0..x.nrows()).rev() {
        if x.row(row).iter().any(|&elem| elem != A::zero()) {
            return Some(row);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use ndarray::{arr2, Array2};

    #[test]
    fn ilalc_empty() {
        let x = Array2::<f32>::eye(0);
        assert!(super::ilalc(&x).is_none());
    }

    #[test]
    fn ilalc_zero() {
        let x = Array2::<f32>::zeros((2, 3));
        assert!(super::ilalc(&x).is_none());
    }

    #[test]
    fn ilalc() {
        let x = arr2(&[[0., 1., 0.], [0., 0., 0.]]);
        assert_eq!(super::ilalc(&x), Some(1));
    }
}
