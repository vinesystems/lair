use criterion::{criterion_group, criterion_main, Criterion};
use lair::decomposition::svd::{svd, svddc};
use ndarray::Array;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::isaac64::Isaac64Rng;

#[cfg(target_os = "macos")]
mod accelerate {
    use std::os::raw::c_char;

    type LapackInt = i32;

    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        fn dgesvd_(
            jobu: *const c_char,
            jobvt: *const c_char,
            m: *const LapackInt,
            n: *const LapackInt,
            a: *mut f64,
            lda: *const LapackInt,
            s: *mut f64,
            u: *mut f64,
            ldu: *const LapackInt,
            vt: *mut f64,
            ldvt: *const LapackInt,
            work: *mut f64,
            lwork: *const LapackInt,
            info: *mut LapackInt,
        );

        fn dgesdd_(
            jobz: *const c_char,
            m: *const LapackInt,
            n: *const LapackInt,
            a: *mut f64,
            lda: *const LapackInt,
            s: *mut f64,
            u: *mut f64,
            ldu: *const LapackInt,
            vt: *mut f64,
            ldvt: *const LapackInt,
            work: *mut f64,
            lwork: *const LapackInt,
            iwork: *mut LapackInt,
            info: *mut LapackInt,
        );
    }

    pub fn gesvd(m: i32, n: i32, a: &mut [f64], s: &mut [f64], u: &mut [f64], vt: &mut [f64]) {
        let jobu = b'A';
        let jobvt = b'A';
        let lda = m;
        let ldu = m;
        let ldvt = n;
        let mut info: i32 = 0;

        // Query optimal work size
        let mut work_query = [0.0f64];
        let lwork_query: i32 = -1;
        unsafe {
            dgesvd_(
                &jobu as *const u8 as *const c_char,
                &jobvt as *const u8 as *const c_char,
                &m,
                &n,
                a.as_mut_ptr(),
                &lda,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                &ldu,
                vt.as_mut_ptr(),
                &ldvt,
                work_query.as_mut_ptr(),
                &lwork_query,
                &mut info,
            );
        }

        let lwork = work_query[0] as i32;
        let mut work = vec![0.0f64; lwork as usize];

        unsafe {
            dgesvd_(
                &jobu as *const u8 as *const c_char,
                &jobvt as *const u8 as *const c_char,
                &m,
                &n,
                a.as_mut_ptr(),
                &lda,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                &ldu,
                vt.as_mut_ptr(),
                &ldvt,
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        assert_eq!(info, 0, "dgesvd_ failed with info = {info}");
    }

    pub fn gesdd(m: i32, n: i32, a: &mut [f64], s: &mut [f64], u: &mut [f64], vt: &mut [f64]) {
        let jobz = b'A';
        let lda = m;
        let ldu = m;
        let ldvt = n;
        let mut info: i32 = 0;
        let min_mn = m.min(n) as usize;
        let mut iwork = vec![0i32; 8 * min_mn];

        // Query optimal work size
        let mut work_query = [0.0f64];
        let lwork_query: i32 = -1;
        unsafe {
            dgesdd_(
                &jobz as *const u8 as *const c_char,
                &m,
                &n,
                a.as_mut_ptr(),
                &lda,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                &ldu,
                vt.as_mut_ptr(),
                &ldvt,
                work_query.as_mut_ptr(),
                &lwork_query,
                iwork.as_mut_ptr(),
                &mut info,
            );
        }

        let lwork = work_query[0] as i32;
        let mut work = vec![0.0f64; lwork as usize];

        unsafe {
            dgesdd_(
                &jobz as *const u8 as *const c_char,
                &m,
                &n,
                a.as_mut_ptr(),
                &lda,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                &ldu,
                vt.as_mut_ptr(),
                &ldvt,
                work.as_mut_ptr(),
                &lwork,
                iwork.as_mut_ptr(),
                &mut info,
            );
        }
        assert_eq!(info, 0, "dgesdd_ failed with info = {info}");
    }
}

fn svd_100(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd");
    let mut rng = Isaac64Rng::seed_from_u64(0);
    let a = Array::random_using((100, 100), Uniform::new(0., 10.).unwrap(), &mut rng);

    group.bench_function("lair", |bencher| {
        bencher.iter(|| {
            let mut a = a.clone();
            let _ = svd(&mut a, true);
        });
    });

    #[cfg(target_os = "macos")]
    group.bench_function("accelerate", |bencher| {
        // Accelerate uses column-major (Fortran) layout
        let a_col_major = a.clone().reversed_axes();
        let a_col_major = a_col_major.as_standard_layout();
        bencher.iter(|| {
            let mut a_data: Vec<f64> = a_col_major.iter().copied().collect();
            let mut s = vec![0.0f64; 100];
            let mut u = vec![0.0f64; 100 * 100];
            let mut vt = vec![0.0f64; 100 * 100];
            accelerate::gesvd(100, 100, &mut a_data, &mut s, &mut u, &mut vt);
        });
    });

    group.finish();
}

fn svddc_100(c: &mut Criterion) {
    let mut group = c.benchmark_group("svddc");
    let mut rng = Isaac64Rng::seed_from_u64(0);
    let a = Array::random_using((100, 100), Uniform::new(0., 10.).unwrap(), &mut rng);

    group.bench_function("lair", |bencher| {
        bencher.iter(|| {
            let mut a = a.clone();
            let _ = svddc(&mut a);
        });
    });

    #[cfg(target_os = "macos")]
    group.bench_function("accelerate", |bencher| {
        // Accelerate uses column-major (Fortran) layout
        let a_col_major = a.clone().reversed_axes();
        let a_col_major = a_col_major.as_standard_layout();
        bencher.iter(|| {
            let mut a_data: Vec<f64> = a_col_major.iter().copied().collect();
            let mut s = vec![0.0f64; 100];
            let mut u = vec![0.0f64; 100 * 100];
            let mut vt = vec![0.0f64; 100 * 100];
            accelerate::gesdd(100, 100, &mut a_data, &mut s, &mut u, &mut vt);
        });
    });

    group.finish();
}

criterion_group!(benches, svd_100, svddc_100);
criterion_main!(benches);
