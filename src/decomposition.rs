//! Matrix decompositions.

pub mod lu;
pub mod qr;
#[cfg(feature = "unstable-svd")]
pub mod svd;
