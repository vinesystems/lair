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

This crate is guaranteed to compile on Rust 1.51 and later.

## License

Copyright 2020-2022 Vine Systems

Licensed under [Apache License, Version 2.0][apache-license] (the "License");
you may not use this crate except in compliance with the License.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See [LICENSE](LICENSE) for
the specific language governing permissions and limitations under the License.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the [Apache-2.0
license][apache-license], shall be licensed as above, without any additional
terms or conditions.

[apache-license]: http://www.apache.org/licenses/LICENSE-2.0
