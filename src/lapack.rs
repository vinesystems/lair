use ndarray::{ArrayBase, DataMut, Dimension};

use crate::{Real, Scalar};

mod bdsdc;
mod bdsqr;
mod gebrd;
mod gelqf;
mod geqrf;
pub mod gesdd;
pub mod gesvd;
mod getrf;
mod getrs;
mod ilal;
pub mod lacpy;
pub mod lange;
pub mod larf;
pub mod larfb;
mod larfg;
pub mod larft;
mod lartg;
mod las2;
pub mod lascl;
pub mod laset;
mod lasq1;
mod lasq2;
mod lasq3;
mod lasq4;
mod lasq5;
mod lasq6;
pub mod lasr;
mod lasv2;
mod laswp;
mod ungbr;
mod unglq;
mod ungqr;
mod ungrq;

pub use geqrf::geqrf;
pub use getrf::getrf;
pub use getrs::getrs;
pub use ilal::{ilalc, ilalr};
pub use larfg::larfg;
pub use laswp::laswp;
pub use unglq::unglq;
use ungqr::ungqr;

fn lacgv<A, S, D>(x: &mut ArrayBase<S, D>)
where
    A: Scalar,
    S: DataMut<Elem = A>,
    D: Dimension,
{
    x.iter_mut().for_each(|x| *x = x.conj());
}

#[allow(dead_code)]
pub fn lapy2<A: Real>(x: A, y: A) -> A {
    (x * x + y * y).sqrt()
}

pub fn lapy3<A: Real>(x: A, y: A, z: A) -> A {
    (x * x + y * y + z * z).sqrt()
}
