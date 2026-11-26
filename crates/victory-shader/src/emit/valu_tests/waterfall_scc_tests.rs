use super::*;
use crate::native::tests::{fspx, fspx_resource};

#[test]
fn a_waterfall_back_edge_cannot_substitute_its_exit_flag_inside_the_loop() {
    let result = try_build_valu_cs(&[
        s_mov_b32(20, 126),
        s_mov_b32(21, 126),
        0xBF06_0000,          // s_cmp_eq_u32 s0, s0; initial SCC is true
        sop2(0x0a, 23, 1, 2), // loop: s_cselect_b32 takes the preceding iteration's SCC
        vop1(0x002, 22, vgpr(1)),
        0x7DA4_0216, // v_cmpx_eq_u32 s22, v1
        0xBE95_4715, // s_andn2_wrexec_b32 s21, s21
        0xBF89_FFFB, // s_cbranch_execnz -> s_cselect
        s_mov_b32(126, 20),
        vop1(0x001, 0, sgpr(23)),
    ]);
    let Err(error) = result else {
        panic!("the loop-carried SCC was accepted")
    };
    assert!(
        matches!(
            error,
            crate::ShaderError::Unsupported {
                what: "a lane mask read as data",
                ..
            }
        ),
        "{error}"
    );
}

/// Each lane saves its own key when it is selected. Returning both the
/// saved key and the tail result keeps the work inside the loop observable.
fn program(wave32: bool, threads: u16, mask: u32, tail: &[u32]) -> HostShader {
    let mut w = vec![s_mov_b32(5, 128)];
    w.extend(s_load_dwordx4(8, 4));
    w.push(0x3602_0080 | mask); // v1 = thread X & mask
    w.push(vop1(1, 6, 193)); // overwritten when this lane's key matches
    if wave32 {
        w.extend([s_mov_b32(20, 126), s_mov_b32(24, 126)]);
    } else {
        w.extend([0xBE94_047E, 0xBE98_047E]); // save exec into two SGPR pairs
    }
    w.extend([
        vop1(0x002, 22, vgpr(1)),                       // read first remaining key
        0x7DA4_0216,                                    // v_cmpx_eq_u32 s22, v1
        vop1(1, 6, sgpr(22)),                           // enabled lanes save the key read this pass
        if wave32 { 0xBE98_4718 } else { 0xBE98_3A18 }, // s_andn2_wrexec
        0xBF89_FFFB,                                    // branch back while any lanes remain
        if wave32 {
            s_mov_b32(126, 20)
        } else {
            0xBEFE_0414
        }, // restore exec
    ]);
    w.extend_from_slice(tail);
    w.extend([
        0x4A04_0017, // v2 = thread X + s23
        0x4A04_0506, // add the key the lane saved in the loop
        0x340E_0082, // v7 = thread X * 4
        0xE070_1000,
        0x8002_0207, // buffer_store_dword v2 at v7
        S_ENDPGM,
    ]);
    let code: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
    let part = fspx(
        &code,
        5 << 1,
        [threads, 1, 1],
        wave32,
        &[fspx_resource(3, 3, 0, 0, 4, 1)],
    );
    let container = crate::dxbc::build(&[(crate::FourCC::ACCC, &part)]);
    super::super::translate(&container).unwrap()
}

#[test]
fn lanes_read_the_waterfall_exit_flag_as_a_condition_and_a_carry() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let hosts: Vec<_> = [("hardware", false), ("warp", true)]
        .into_iter()
        .filter_map(|(label, warp)| Host::new(warp).ok().map(|host| (label, host)))
        .collect();
    assert!(
        !hosts.is_empty(),
        "neither hardware nor WARP SM 6.6 device available"
    );
    let tails: [(&str, &[u32], u32); 3] = [
        ("s_cselect_b32", &[sop2(0x0a, 23, 145, 157)], 29), // s23 = SCC ? 17 : 29
        ("s_addc_u32", &[sop2(0x04, 23, 169, 133)], 46),    // s23 = 41 + 5 + SCC
        (
            "s_cbranch_scc1",
            &[s_mov_b32(23, 145), 0xBF85_0001, s_mov_b32(23, 157)],
            29,
        ),
    ];
    let mut validator =
        validator::Validator::load(validator::Validator::default_path().as_deref()).ok();
    for (wave32, threads) in [(true, 32u16), (false, 64)] {
        for mask in [0u32, 1, 3, 7] {
            for (consumer, tail, expected) in tails {
                let shader = program(wave32, threads, mask, tail);
                if let Some(v) = &mut validator {
                    v.validate(&mut shader.dxil.clone()).unwrap();
                }
                for (backend, host) in &hosts {
                    let (root, pso) = host.compute_pipeline(&shader).unwrap();
                    let data = vec![0; shader.user_data_dwords().max(1) as usize];
                    let out = host
                        .dispatch(
                            &root,
                            &pso,
                            &Dispatch {
                                roots: vec![(0, Root::Constants(data))],
                                views: &[View::RawUav(u32::from(threads))],
                                samplers: &[],
                                groups: [1, 1, 1],
                            },
                        )
                        .unwrap();
                    for lane in 0..u32::from(threads) {
                        assert_eq!(
                            out[0][lane as usize],
                            expected + lane + (lane & mask),
                            "{backend} wave{threads} {consumer} mask={mask} lane={lane}"
                        );
                    }
                }
            }
        }
    }
}
