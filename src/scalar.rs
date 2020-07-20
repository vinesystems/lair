use num_complex::Complex;
use num_traits::{Float, One, Zero};
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A trait for real and complex numbers.
pub trait Scalar:
    Copy
    + Debug
    + Display
    + PartialEq
    + Zero
    + One
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Sum
{
    type Real: Real + Into<Self>;

    fn re(&self) -> Self::Real;
    fn im(&self) -> Self::Real;
    fn conj(&self) -> Self;
    fn norm_sqr(&self) -> Self::Real;
}

impl Scalar for f32 {
    type Real = Self;

    #[inline]
    fn re(&self) -> Self::Real {
        *self
    }

    #[inline]
    fn im(&self) -> Self::Real {
        0.
    }

    #[inline]
    fn conj(&self) -> Self {
        *self
    }

    #[inline]
    fn norm_sqr(&self) -> Self::Real {
        *self * *self
    }
}

impl Scalar for f64 {
    type Real = Self;

    #[inline]
    fn re(&self) -> Self::Real {
        *self
    }

    #[inline]
    fn im(&self) -> Self::Real {
        0.
    }

    #[inline]
    fn conj(&self) -> Self {
        *self
    }

    #[inline]
    fn norm_sqr(&self) -> Self::Real {
        *self * *self
    }
}

impl Scalar for Complex<f32> {
    type Real = f32;

    #[inline]
    fn re(&self) -> Self::Real {
        self.re
    }

    #[inline]
    fn im(&self) -> Self::Real {
        self.im
    }

    #[inline]
    fn conj(&self) -> Self {
        self.conj()
    }

    #[inline]
    fn norm_sqr(&self) -> Self::Real {
        self.norm_sqr()
    }
}

impl Scalar for Complex<f64> {
    type Real = f64;

    #[inline]
    fn re(&self) -> Self::Real {
        self.re
    }

    #[inline]
    fn im(&self) -> Self::Real {
        self.im
    }

    #[inline]
    fn conj(&self) -> Self {
        self.conj()
    }

    #[inline]
    fn norm_sqr(&self) -> Self::Real {
        self.norm_sqr()
    }
}

/// A trait for real numbers.
pub trait Real: Float + Zero + One + AddAssign + SubAssign + MulAssign + DivAssign + Sum {
    /// Returns a number composed of the magnitude of `self` and the sign of
    /// `sign`.
    fn copysign(self, sign: Self) -> Self;

    /// Relative machine precision.
    #[inline]
    #[must_use]
    fn eps() -> Self {
        Self::epsilon() / (Self::one() + Self::one())
    }

    /// Safe minimum, such that its reciprocal does not overflow.
    #[inline]
    #[must_use]
    fn sfmin() -> Self {
        Self::min_positive_value()
    }
}

impl Real for f32 {
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        self.copysign(sign)
    }
}

impl Real for f64 {
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        self.copysign(sign)
    }
}
