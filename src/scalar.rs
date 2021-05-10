use ndarray::ScalarOperand;
use num_complex::Complex;
use num_traits::{Float, FromPrimitive, NumAssign, One, Zero};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display, LowerExp, UpperExp};
use std::iter::{Product, Sum};
use std::ops::Neg;

macro_rules! trait_scalar_body {
    () => {
        fn re(&self) -> Self::Real;
        fn im(&self) -> Self::Real;
        fn conj(&self) -> Self;
        fn abs(&self) -> Self::Real;
        fn square(&self) -> Self::Real;

        fn sqrt(self) -> Self;
        fn exp(self) -> Self;
        fn ln(self) -> Self;

        fn sin(self) -> Self;
        fn cos(self) -> Self;
        fn tan(self) -> Self;
        fn sinh(self) -> Self;
        fn cosh(self) -> Self;
        fn tanh(self) -> Self;
        fn asin(self) -> Self;
        fn acos(self) -> Self;
        fn atan(self) -> Self;
        fn asinh(self) -> Self;
        fn acosh(self) -> Self;
        fn atanh(self) -> Self;
    };
}

macro_rules! impl_scalar_t_body {
    () => {
        #[inline]
        fn re(&self) -> Self::Real {
            *self
        }

        #[inline]
        fn im(&self) -> Self::Real {
            T::zero()
        }

        #[inline]
        fn conj(&self) -> Self {
            *self
        }

        #[inline]
        fn abs(&self) -> Self::Real {
            Self::abs(*self)
        }

        #[inline]
        fn square(&self) -> Self::Real {
            *self * *self
        }

        #[inline]
        fn sqrt(self) -> Self {
            self.sqrt()
        }

        #[inline]
        fn exp(self) -> Self {
            self.exp()
        }

        #[inline]
        fn ln(self) -> Self {
            self.ln()
        }

        #[inline]
        fn sin(self) -> Self {
            self.sin()
        }

        #[inline]
        fn cos(self) -> Self {
            self.cos()
        }

        #[inline]
        fn tan(self) -> Self {
            self.tan()
        }

        #[inline]
        fn sinh(self) -> Self {
            self.sinh()
        }

        #[inline]
        fn cosh(self) -> Self {
            self.cosh()
        }

        #[inline]
        fn tanh(self) -> Self {
            self.tanh()
        }

        #[inline]
        fn asin(self) -> Self {
            self.asin()
        }

        #[inline]
        fn acos(self) -> Self {
            self.acos()
        }

        #[inline]
        fn atan(self) -> Self {
            self.atan()
        }

        #[inline]
        fn asinh(self) -> Self {
            self.asinh()
        }

        #[inline]
        fn acosh(self) -> Self {
            self.acosh()
        }

        #[inline]
        fn atanh(self) -> Self {
            self.atanh()
        }
    };
}

macro_rules! impl_scalar_complex_body {
    () => {
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
        fn abs(&self) -> Self::Real {
            self.norm()
        }

        #[inline]
        fn square(&self) -> Self::Real {
            self.norm_sqr()
        }

        #[inline]
        fn sqrt(self) -> Self {
            self.sqrt()
        }

        #[inline]
        fn exp(self) -> Self {
            self.exp()
        }

        #[inline]
        fn ln(self) -> Self {
            self.ln()
        }

        #[inline]
        fn sin(self) -> Self {
            self.sin()
        }

        #[inline]
        fn cos(self) -> Self {
            self.cos()
        }

        #[inline]
        fn tan(self) -> Self {
            self.tan()
        }

        #[inline]
        fn sinh(self) -> Self {
            self.sinh()
        }

        #[inline]
        fn cosh(self) -> Self {
            self.cosh()
        }

        #[inline]
        fn tanh(self) -> Self {
            self.tanh()
        }

        #[inline]
        fn asin(self) -> Self {
            self.asin()
        }

        #[inline]
        fn acos(self) -> Self {
            self.acos()
        }

        #[inline]
        fn atan(self) -> Self {
            self.atan()
        }

        #[inline]
        fn asinh(self) -> Self {
            self.asinh()
        }

        #[inline]
        fn acosh(self) -> Self {
            self.acosh()
        }

        #[inline]
        fn atanh(self) -> Self {
            self.atanh()
        }
    };
}

#[cfg(feature = "serde")]
/// A trait for real and complex numbers.
pub trait Scalar:
    Copy
    + Debug
    + Display
    + LowerExp
    + UpperExp
    + PartialEq
    + Neg<Output = Self>
    + NumAssign
    + FromPrimitive
    + Sum
    + Product
    + ScalarOperand
    + Serialize
    + for<'de> Deserialize<'de>
{
    type Real: Real + Into<Self> + FromPrimitive + Serialize + for<'de> Deserialize<'de>;
    trait_scalar_body!();
}

#[cfg(feature = "serde")]
impl<T> Scalar for T
where
    T: Debug
        + Display
        + LowerExp
        + UpperExp
        + NumAssign
        + FromPrimitive
        + Sum
        + Product
        + Real
        + ScalarOperand
        + Serialize
        + for<'de> Deserialize<'de>,
{
    type Real = Self;

    impl_scalar_t_body!();
}

#[cfg(feature = "serde")]
impl<T> Scalar for Complex<T>
where
    T: Copy
        + Debug
        + Display
        + LowerExp
        + UpperExp
        + Neg<Output = T>
        + NumAssign
        + FromPrimitive
        + Real
        + Serialize
        + for<'de> Deserialize<'de>,
    Complex<T>: ScalarOperand + Serialize + for<'de> Deserialize<'de>,
{
    type Real = T;

    impl_scalar_complex_body!();
}

#[cfg(not(feature = "serde"))]
/// A trait for real and complex numbers.
pub trait Scalar:
    Copy
    + Debug
    + Display
    + LowerExp
    + UpperExp
    + PartialEq
    + Neg<Output = Self>
    + NumAssign
    + FromPrimitive
    + Sum
    + Product
    + ScalarOperand
{
    type Real: Real + Into<Self> + FromPrimitive;
    trait_scalar_body!();
}

#[cfg(not(feature = "serde"))]
impl<T> Scalar for T
where
    T: Debug
        + Display
        + LowerExp
        + UpperExp
        + NumAssign
        + FromPrimitive
        + Sum
        + Product
        + Real
        + ScalarOperand,
{
    type Real = Self;

    impl_scalar_t_body!();
}

#[cfg(not(feature = "serde"))]
impl<T> Scalar for Complex<T>
where
    T: Copy
        + Debug
        + Display
        + LowerExp
        + UpperExp
        + Neg<Output = T>
        + NumAssign
        + FromPrimitive
        + Real,
    Complex<T>: ScalarOperand,
{
    type Real = T;

    impl_scalar_complex_body!();
}

/// A trait for real numbers.
pub trait Real: Float + Zero + One + NumAssign + Sum {
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
