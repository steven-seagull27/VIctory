// Translate invented GFX10 VINTRP pairs through the public VIctory API on 63bcca43.
use std::{env, fs};
use victory_shader::native::{PsInput, StageIo, Usage};
use victory_shader::{Container, HwStage, NativeProgram, emit, io, ir, resource, structure};

fn main() {
    let args: Vec<_> = env::args().skip(1).collect();
    let signature = fs::read(&args[0]).expect("read reference signature");
    let input = Container::parse(&signature).expect("parse signature fixture");
    let mut words = Vec::new();
    // Row 0 has (TEXCOORD4.xy, TEXCOORD1.z, TEXCOORD7.w) with perspective-center.
    // Row 1 has TEXCOORD9.x with linear-center. VGPRs v0/v1 and v2/v3
    // receive these barycentric pairs respectively.
    for (dst, attr, chan, bary) in [
        (4u32, 0u32, 1u32, 0u32),
        (5, 0, 2, 0),
        (6, 0, 3, 0),
        (7, 1, 0, 2),
    ] {
        let p1 = 0xc8000000 | (dst << 18) | (attr << 10) | (chan << 8);
        words.push(p1 | bary);
        words.push(p1 | 0x10000 | (bary + 1));
    }
    words.extend([0xf800180f, 0x07060504, 0xbf810000]); // exp mrt0 xyzw v4..7, done vm; end
    let attr = |row, mask| PsInput {
        usage: Usage::Generic,
        row,
        mask,
        attr: row,
        flat: false,
        fp16: false,
        pack2: false,
        row2: 0,
    };
    let prog = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 8,
        layout: Vec::new(),
        used: Vec::new(),
        code: words.into_iter().flat_map(u32::to_le_bytes).collect(),
        io: StageIo::Ps {
            inputs: vec![attr(0, 15), attr(1, 1)],
            input_addr: (1 << 1) | (1 << 5),
            db_shader_control: 0,
            patches: Vec::new(),
        },
    };
    let p = ir::build(&prog.patched_code(), prog.wave64, prog.vgprs).expect("IR");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &prog.layout, prog.stage, Some(&st)).expect("resources");
    let interface = io::raise(&p, &prog).expect("inputs and outputs");
    let host = emit::emit(&input, &prog, &p, &st, &interface, &res).expect("emit DXIL");
    fs::write(&args[1], host.dxil).expect("save translation");
    println!(
        "victory-shader source fingerprint {}",
        victory_shader::SOURCE_HASH
    );
}
