use criterion::{criterion_group, criterion_main, Criterion};
use lair::decomposition::LUFactorized;
use ndarray::Array;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::isaac64::Isaac64Rng;
use std::convert::TryFrom;

fn getrs_100(c: &mut Criterion) {
    let mut rng = Isaac64Rng::seed_from_u64(0);
    let a = Array::random_using((100, 100), Uniform::new(0., 10.), &mut rng);
    let b = Array::random_using(100, Uniform::new(0., 10.), &mut rng);
    let lu = LUFactorized::try_from(a).expect("non-singular");
    c.bench_function("getrs", |bencher| {
        bencher.iter(|| {
            lu.solve(&b);
        })
    });
}

criterion_group!(benches, getrs_100);
criterion_main!(benches);
