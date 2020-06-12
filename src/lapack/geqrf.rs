use ndarray::{Array2, ArrayBase, Data, DataMut, Ix1, Ix2, LinalgScalar, NdFloat};
use std::ops::SubAssign;

pub fn geqrf<A, S>(a: &mut ArrayBase<S, Ix2>) -> Array2<A>
where
    A: NdFloat + std::iter::Sum + LinalgScalar + SubAssign,
    S: DataMut<Elem = A>,
{
    let mut r = Array2::<A>::zeros((a.nrows(), a.ncols()));
    if a.ncols() < 1 {
        return r;
    }
    let len = euclidean(&a.column(0));
    r.row_mut(0)[0] = len;
    for ai in a.column_mut(0) {
        *ai /= len;
    }

    for j in 1..a.ncols() {
        for i in 0..j {
            let rij: A = a
                .column(i)
                .iter()
                .zip(a.column(j).iter())
                .map(|(ai, aj)| *ai * *aj)
                .sum();

            for elem in 0..a.nrows() {
                let delta = a[(elem, i)] * rij;
                a[(elem, j)] -= delta;
            }
            r[(i, j)] = rij;
        }
        let len = euclidean(&a.column(j));
        a.column_mut(j).mapv_inplace(|val| val / len);
        r[(j, j)] = len;
    }
    r
}

fn euclidean<A, S>(v: &ArrayBase<S, Ix1>) -> A
where
    A: NdFloat + LinalgScalar + std::iter::Sum,
    S: Data<Elem = A>,
{
    let len_sq: A = v.iter().map(|vi| *vi * *vi).sum();
    len_sq.sqrt()
}

#[cfg(test)]
mod test {
    use ndarray::{arr1, arr2};

    #[test]
    fn euclidean() {
        let a = arr1(&[3_f64, 4_f64]);
        assert_eq!(5_f64, super::euclidean(&a));
    }

    #[test]
    fn geqrf_3() {
        let mut a = arr2(&[
            [1_f64, 2_f64, 4_f64],
            [0_f64, 0_f64, 5_f64],
            [0_f64, 3_f64, 6_f64],
        ]);
        let q = arr2(&[
            [1_f64, 0_f64, 0_f64],
            [0_f64, 0_f64, 1_f64],
            [0_f64, 1_f64, 0_f64],
        ]);
        let r = arr2(&[
            [1_f64, 2_f64, 4_f64],
            [0_f64, 3_f64, 6_f64],
            [0_f64, 0_f64, 5_f64],
        ]);

        let r_ret = super::geqrf(&mut a);
        assert_eq!(r_ret, r);
        assert_eq!(a, q);
    }
}
