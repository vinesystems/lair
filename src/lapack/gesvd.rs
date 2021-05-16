use super::{gebrd, geqrf, lacpy, lascl, laset, ungbr, ungqr};
use crate::{Real, Scalar};
use ndarray::{s, Array2, ArrayBase, DataMut, Ix2};
use num_traits::{Float, ToPrimitive, Zero};
use std::ops::{Div, MulAssign};

#[allow(dead_code)]
pub fn gesvd<A, S>(a: &mut ArrayBase<S, Ix2>)
where
    A: Scalar + Div<<A as Scalar>::Real, Output = A> + MulAssign<<A as Scalar>::Real>,
    S: DataMut<Elem = A>,
{
    if a.is_empty() {
        return;
    }

    let small_num = A::Real::sfmin().sqrt() / A::Real::prec();
    let large_num = small_num.recip();

    let a_norm = super::lange::maxabs(a);
    let _is_scaled = if a_norm > A::Real::zero() && a_norm < small_num {
        lascl::full(a_norm, small_num, a);
        true
    } else if a_norm > large_num {
        lascl::full(a_norm, large_num, a);
        true
    } else {
        false
    };

    if a.nrows() >= a.ncols() {
        inner_tall(a);
    } else {
        inner_wide(a);
    }
    unimplemented!()
}

fn inner_tall<A, S>(a: &mut ArrayBase<S, Ix2>)
where
    A: Scalar + Div<<A as Scalar>::Real, Output = A> + MulAssign<<A as Scalar>::Real>,
    S: DataMut<Elem = A>,
{
    let mut u = Array2::<A>::zeros((a.nrows(), a.nrows()));
    let mut vt = Array2::<A>::zeros((a.ncols(), a.ncols()));
    if a.nrows() >= svd_crossover_point(a.ncols()) {
        let tau = geqrf(a);
        lacpy::lower(a, &mut u);
        ungqr(&mut u, &tau);
        let mut u_work = Array2::<A>::zeros((a.ncols(), a.ncols()));
        laset::lower_zero(&mut u_work.slice_mut(s![1.., ..u_work.ncols() - 1]));
        let (_, _, tau_q, tau_p) = gebrd::tall(&mut u_work);
        lacpy::upper(a, &mut vt);
        debug_assert_eq!(u_work.nrows(), tau_q.len());
        ungbr::q_tall(&mut u_work, &tau_q);
        debug_assert_eq!(vt.nrows(), tau_p.len());
        ungbr::p_square(&mut vt, &tau_p);
        todo!("bdsqr")
    } else {
        let (_, _, tau_q, tau_p) = gebrd::tall(a);
        lacpy::lower(a, &mut u);
        ungbr::q_tall(&mut u, &tau_q);
        lacpy::upper(&a.slice(s![..a.ncols(), ..]), &mut vt);
        ungbr::p_square(&mut vt, &tau_p);
        todo!("bdsqr")
    }
}

fn inner_wide<A, S>(a: &mut ArrayBase<S, Ix2>)
where
    A: Scalar + Div<<A as Scalar>::Real, Output = A> + MulAssign<<A as Scalar>::Real>,
    S: DataMut<Elem = A>,
{
    if a.ncols() >= svd_crossover_point(a.nrows()) {
        todo!("gelqf")
    } else {
        todo!("ungbr, bdsqr")
    }
}

fn svd_crossover_point(n: usize) -> usize {
    (n.to_f32().expect("never fails") * 1.6)
        .to_usize()
        .unwrap_or(usize::MAX)
}
