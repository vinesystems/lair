#![cfg(feature = "bench-lapack")]

use criterion::{criterion_group, criterion_main, Criterion};
use lair::bench_lapack::gebrd_tall;
use ndarray::Array;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::isaac64::Isaac64Rng;

#[cfg(target_os = "macos")]
mod accelerate {
    type LapackInt = i32;

    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        fn dgebrd_(
            m: *const LapackInt,
            n: *const LapackInt,
            a: *mut f64,
            lda: *const LapackInt,
            d: *mut f64,
            e: *mut f64,
            tauq: *mut f64,
            taup: *mut f64,
            work: *mut f64,
            lwork: *const LapackInt,
            info: *mut LapackInt,
        );
    }

    pub fn gebrd(
        m: i32,
        n: i32,
        a: &mut [f64],
        d: &mut [f64],
        e: &mut [f64],
        tauq: &mut [f64],
        taup: &mut [f64],
    ) {
        let lda = m;
        let mut info: i32 = 0;

        // Query optimal work size
        let mut work_query = [0.0f64];
        let lwork_query: i32 = -1;
        unsafe {
            dgebrd_(
                &m,
                &n,
                a.as_mut_ptr(),
                &lda,
                d.as_mut_ptr(),
                e.as_mut_ptr(),
                tauq.as_mut_ptr(),
                taup.as_mut_ptr(),
                work_query.as_mut_ptr(),
                &lwork_query,
                &mut info,
            );
        }

        let lwork = work_query[0] as i32;
        let mut work = vec![0.0f64; lwork as usize];

        unsafe {
            dgebrd_(
                &m,
                &n,
                a.as_mut_ptr(),
                &lda,
                d.as_mut_ptr(),
                e.as_mut_ptr(),
                tauq.as_mut_ptr(),
                taup.as_mut_ptr(),
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        assert_eq!(info, 0, "dgebrd_ failed with info = {info}");
    }
}

fn gebrd_200x100(c: &mut Criterion) {
    let mut group = c.benchmark_group("gebrd");
    let mut rng = Isaac64Rng::seed_from_u64(0);
    let m: i32 = 200;
    let n: i32 = 100;
    let a = Array::random_using(
        (m as usize, n as usize),
        Uniform::new(0., 10.).unwrap(),
        &mut rng,
    );

    group.bench_function("lair", |bencher| {
        bencher.iter(|| {
            let mut a = a.clone();
            let _ = gebrd_tall(&mut a);
        });
    });

    #[cfg(target_os = "macos")]
    group.bench_function("accelerate", |bencher| {
        // Accelerate uses column-major (Fortran) layout
        let a_col_major = a.clone().reversed_axes();
        let a_col_major = a_col_major.as_standard_layout();
        bencher.iter(|| {
            let mut a_data: Vec<f64> = a_col_major.iter().copied().collect();
            let mut d = vec![0.0f64; n as usize];
            let mut e = vec![0.0f64; (n - 1) as usize];
            let mut tauq = vec![0.0f64; n as usize];
            let mut taup = vec![0.0f64; n as usize];
            accelerate::gebrd(m, n, &mut a_data, &mut d, &mut e, &mut tauq, &mut taup);
        });
    });

    group.finish();
}

criterion_group!(benches, gebrd_200x100);
criterion_main!(benches);
