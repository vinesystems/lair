use criterion::{criterion_group, criterion_main, Criterion};
use lair::decomposition::QRFactorized;
use ndarray::Array;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::isaac64::Isaac64Rng;

fn geqrf_100(c: &mut Criterion) {
    let mut rng = Isaac64Rng::seed_from_u64(0);
    let a = Array::random_using((100, 100), Uniform::new(0., 10.), &mut rng);
    c.bench_function("geqrf", |bencher| {
        bencher.iter(|| {
            let a = a.clone();
            QRFactorized::from(a);
        })
    });
}

criterion_group!(benches, geqrf_100);
criterion_main!(benches);
