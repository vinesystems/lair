use criterion::{criterion_group, criterion_main, Criterion};
use lair::decomposition::QRFactorized;
use ndarray::arr2;

fn criterion_benchmark(c: &mut Criterion) {
    let a = arr2(&[
        [1_f64, 2_f64, 4_f64],
        [0_f64, 0_f64, 5_f64],
        [0_f64, 3_f64, 6_f64],
    ]);
    c.bench_function("geqrf(Gram-Schmidt)", |b| {
        b.iter(|| {
            let a = a.clone();
            QRFactorized::from(a);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
