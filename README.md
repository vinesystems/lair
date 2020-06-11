# Lair: Linear Algebra in Rust

Lair implements linear algebra routines in pure Rust. It uses [ndarray] as its
matrix representation.

[ndarray]: https://github.com/bluss/ndarray

[![Coverage Status](https://codecov.io/gh/vinesystems/lair/branch/master/graphs/badge.svg)](https://codecov.io/gh/vinesystems/lair)

## Requirements

* Rust ≥ 1.37

## Features

Lair is still in an early stage, and provides only a limited set of functions,
including the followings:

* An equation solver for a system of linear scalar equations
* Matrix builders for special matrices such as circulant and companion matrices

## License

Licensed under [Apache License, Version 2.0][apache-license]
([LICENSE](LICENSE)).

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the [Apache-2.0
license][apache-license], shall be licensed as above, without any additional
terms or conditions.

[apache-license]: http://www.apache.org/licenses/LICENSE-2.0
