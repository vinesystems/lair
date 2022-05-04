# Changelog

This file documents recent notable changes to this project. The format of this
file is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2022-05-04

### Removed

* `Real::copysign` has been replaced with num-traits's `Float::copysign`.

## [0.5.0] - 2021-05-10

### Added

* A new `serialization` feature enables serialization/deserialization support
  for `Scalar`.
* `Scalar` has `sqrt`, `exp`, `ln` and trigonometry functions.
* `LuFactorized::into_pl` returns P * L of LU decomposition.

### Changed

* Requires Rust 1.51 or later.

### Removed

* No longer re-exports `Float`, `One`, and `Zero`; use num-traits directly.

## [0.4.0] - 2021-03-28

### Added

* `Scalar::abs` returns the absolute value.

### Changed

* Requires Rust 1.49 or later.
* Updated ndarray to 0.15.
* Follows Rust naming conventions:
  - `LUFactorized` is now `lu::Factorized`.
  - `QRFactorized` is now `qr::Factorized`.
* `Scalar::norm_sqr` has been renaemd `Scalar::square`.

## [0.3.0] - 2021-02-16

### Changed

* Updated ndarray to 0.14.
* Removed dependency on cauchy.  `cauchy::Scalar` has been replaced with a
  simpler implementation.
* LU factorization with a singular matrix does not return an error.

## [0.2.0] - 2020-06-26

### Changed

* Decompositions work for complex matrices.
* `circulant` and `companion` can build complex matrices.

## [0.1.1] - 2020-06-23

### Added

* LU decomposition.
* QR decomposition.
* Support for complex numbers.

## [0.1.0] - 2020-06-11

### Added

* An equation solver for a system of linear scalar equations.
* A circulent matrix builder.
* A companion matrix builder.

[0.6.0]: https://github.com/vinesystems/lair/compare/0.5.0...0.6.0
[0.5.0]: https://github.com/vinesystems/lair/compare/0.4.0...0.5.0
[0.4.0]: https://github.com/vinesystems/lair/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/vinesystems/lair/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/vinesystems/lair/compare/0.1.1...0.2.0
[0.1.1]: https://github.com/vinesystems/lair/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/vinesystems/lair/tree/0.1.0
