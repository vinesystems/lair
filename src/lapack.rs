use crate::Real;

mod geqrf;
mod getrf;
mod getrs;
mod ilal;
pub(crate) mod larf;
mod larfg;
mod lartg;
mod las2;
pub(crate) mod lascl;
mod lasq;
mod laswp;
mod orgrq;

pub(crate) use geqrf::geqrf;
pub(crate) use getrf::getrf;
pub(crate) use getrs::getrs;
pub(crate) use ilal::{ilalc, ilalr};
pub(crate) use larfg::larfg;
pub(crate) use laswp::laswp;

#[allow(dead_code)]
pub(crate) fn lapy2<A: Real>(x: A, y: A) -> A {
    (x * x + y * y).sqrt()
}

pub(crate) fn lapy3<A: Real>(x: A, y: A, z: A) -> A {
    (x * x + y * y + z * z).sqrt()
}
