//! Batch the ordinary translation entry point on public synthetic compressed compute shaders.
use super::{HostShader, Options, container, translate_with};
use crate::native::tests::compute_with_indexed_uav;
use std::path::PathBuf;
use std::time::Instant;

/// The indexed buffer and VALU chain from #2942, with an optional scalar
/// diamond every eight loads. Both sides feed the next dependent load.
fn shader(id: usize) -> Vec<u8> {
    let branched = id % 2 == 1;
    let rank = id / 2;
    let repeats = 8 + rank * 12 + rank.pow(3) / 24;
    let mut w = vec![0xBE84_0300u32, 0xBE85_0380, 0xF408_0202, 0xFA00_0000];
    for i in 0..repeats {
        w.extend([
            0xE030_2000,
            0x8002_0200,
            0x0602_0502,
            0x1002_0301,
            0x4A00_0101,
        ]);
        for j in 0..7 {
            let dst = 3 + ((i + j) & 3) as u32;
            w.push(0x0600_0301 | (dst << 17));
        }
        if branched && i % 8 == 0 {
            w.extend([
                0xBF06_0002 | ((128 + ((i / 8) & 31)) as u32) << 8, // group X == k
                0xBF85_0002, // s_cbranch_scc1 to the second add
                0x4A00_0081, // v0 += 1
                0xBF82_0001, // jump to the join
                0x4A00_0082, // v0 += 2
            ]);
        }
    }
    w.extend([0xE070_2000, 0x8002_0000, 0xBF81_0000]);
    let code: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
    container::build(&[(b"DXIL", &compute_with_indexed_uav(&code))])
}

fn translate(id: usize, code: &[u8]) -> HostShader {
    translate_with(
        code,
        Options {
            view_marks: id / 2 % 2 == 1,
        },
    )
    .unwrap()
}

/// Sixteen VGPRs cross each runtime diamond and all feed one UAV store.
/// The two paths add different values to every live register.
fn join_shader(id: usize) -> Vec<u8> {
    let diamonds = 4 + 2 * id;
    let mut w = vec![0xBE84_0300u32, 0xBE85_0380, 0xF408_0202, 0xFA00_0000];
    for r in 8..24u32 {
        w.push(0x4A00_0000 | r << 17 | (128 + r)); // v[r] = tid + r
    }
    for i in 0..diamonds {
        w.push(0xBF06_0002 | ((128 + (i & 31)) as u32) << 8);
        w.push(0xBF85_0011); // skip sixteen adds and the jump to the join
        for r in 8..24u32 {
            w.push(0x4A00_0000 | r << 17 | r << 9 | (128 + r));
        }
        w.push(0xBF82_0010);
        for r in 8..24u32 {
            w.push(0x4A00_0000 | r << 17 | r << 9 | (152 - r));
        }
    }
    w.push(0x7E02_0280); // v1 = 0
    for r in 8..24u32 {
        w.push(0x4A02_0300 | r); // v1 += v[r]
    }
    w.extend([0xE070_2000, 0x8002_0100, 0xBF81_0000]); // store v1 at tid
    let code: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
    container::build(&[(b"DXIL", &compute_with_indexed_uav(&code))])
}

#[test]
#[ignore = "CPU throughput diagnostic; run with --release --ignored --nocapture"]
fn translate_public_compute_batch() {
    let generate = if std::env::var("VICTORY_BENCH_CORPUS").as_deref() == Ok("joins") {
        join_shader
    } else {
        shader
    };
    let corpus: Vec<_> = (0..64).map(generate).collect();
    let workers: usize = std::env::var("VICTORY_BENCH_WORKERS")
        .map_or(Ok(1), |s| s.parse())
        .unwrap();
    assert!((1..=corpus.len()).contains(&workers));
    if let Some(dir) = std::env::var_os("VICTORY_BENCH_OUTPUT").map(PathBuf::from) {
        std::fs::create_dir_all(&dir).unwrap();
        #[cfg(windows)]
        let mut validator = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        )
        .expect("Windows SDK dxil.dll");
        for (id, code) in corpus.iter().enumerate() {
            let mut out = translate(id, code);
            let dxil = std::mem::take(&mut out.dxil);
            #[cfg(windows)]
            validator.validate(&mut dxil.clone()).unwrap();
            std::fs::write(dir.join(format!("{id}.dxil")), dxil).unwrap();
            std::fs::write(dir.join(format!("{id}.bindings")), format!("{out:?}")).unwrap();
        }
    }
    let run = || {
        std::thread::scope(|scope| {
            for worker in 0..workers {
                let shaders = &corpus;
                scope.spawn(move || {
                    for id in (worker..shaders.len()).step_by(workers) {
                        std::hint::black_box(translate(id, &shaders[id]));
                    }
                });
            }
        })
    };
    run();
    for _ in 0..3 {
        let start = Instant::now();
        run();
        eprintln!(
            "batch of {} on {workers} workers: {:.3} ms",
            corpus.len(),
            start.elapsed().as_secs_f64() * 1000.
        );
    }
}
