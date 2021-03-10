use crate::Real;

mod geqrf;
mod getrf;
mod getrs;
mod ilal;
pub mod larf;
mod larfg;
mod lartg;
mod las2;
pub mod lascl;
mod lasq;
mod laswp;
mod orgrq;

pub use geqrf::geqrf;
pub use getrf::getrf;
pub use getrs::getrs;
pub use ilal::{ilalc, ilalr};
pub use larfg::larfg;
pub use laswp::laswp;

#[allow(dead_code)]
pub fn lapy2<A: Real>(x: A, y: A) -> A {
    (x * x + y * y).sqrt()
}

pub fn lapy3<A: Real>(x: A, y: A, z: A) -> A {
    (x * x + y * y + z * z).sqrt()
}
