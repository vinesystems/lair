# Lair: Linear Algebra in Rust

Lair implements linear algebra routines in pure Rust. It uses [ndarray] as its
matrix representation.

[ndarray]: https://github.com/bluss/ndarray

[![crates.io](https://img.shields.io/crates/v/lair)](https://crates.io/crates/lair)
[![Documentation](https://docs.rs/lair/badge.svg)](https://docs.rs/lair)
[![Coverage Status](https://codecov.io/gh/vinesystems/lair/branch/master/graphs/badge.svg)](https://codecov.io/gh/vinesystems/lair)

## Features

Lair is still in an early stage, and provides only a limited set of functions,
including the followings:

* LU and QR decompositions
* An equation solver for a system of linear scalar equations
* Matrix builders for special matrices such as circulant and companion matrices

## Minimum Supported Rust Version

This crate is guaranteed to compile on Rust 1.43 and later.

## License

Licensed under [Apache License, Version 2.0][apache-license]
([LICENSE](LICENSE)).

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the [Apache-2.0
license][apache-license], shall be licensed as above, without any additional
terms or conditions.

[apache-license]: http://www.apache.org/licenses/LICENSE-2.0
