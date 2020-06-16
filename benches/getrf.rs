use criterion::{criterion_group, criterion_main, Criterion};
use lair::decomposition::LUFactorized;
use ndarray::Array;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::isaac64::Isaac64Rng;
use std::convert::TryFrom;

fn getrf_100(c: &mut Criterion) {
    let mut rng = Isaac64Rng::seed_from_u64(0);
    let a = Array::random_using((100, 100), Uniform::new(0., 10.), &mut rng);
    c.bench_function("getrf", |bencher| {
        bencher.iter(|| {
            let a = a.clone();
            LUFactorized::try_from(a).expect("non-singular");
        })
    });
}

criterion_group!(benches, getrf_100);
criterion_main!(benches);
