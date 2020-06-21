use criterion::{criterion_group, criterion_main, Criterion};
use lair::decomposition::LUFactorized;
use ndarray::Array;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::isaac64::Isaac64Rng;
use std::convert::TryFrom;

fn getrf_100(c: &mut Criterion) {
    let mut group = c.benchmark_group("lu");
    let mut rng = Isaac64Rng::seed_from_u64(0);
    let a_row_major = Array::random_using((100, 100), Uniform::new(0., 10.), &mut rng);
    let a_tmp = a_row_major.clone().reversed_axes();
    let a_col_major = a_tmp.as_standard_layout().reversed_axes();
    assert!(a_row_major.is_standard_layout());
    assert!(!a_col_major.is_standard_layout());
    assert_eq!(a_row_major, a_col_major);
    group.bench_function("row-major", |bencher| {
        bencher.iter(|| {
            let a = a_row_major.clone();
            LUFactorized::try_from(a).expect("non-singular");
        })
    });
    group.bench_function("col-major", |bencher| {
        bencher.iter(|| {
            let mut a = a_col_major.clone();
            a.swap_axes(0, 1);
            LUFactorized::try_from(a).expect("non-singular");
        })
    });
}

criterion_group!(benches, getrf_100);
criterion_main!(benches);
