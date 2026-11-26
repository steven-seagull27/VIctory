//! Private offline translation manifests. Match a full input container and its
//! native program ordinal, including containers whose signature parts differ.
use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result, ensure};
use blake3::Hash;
use victory_shader::emit::{HostShader, Options};
use victory_shader::{Container, FourCC, NativeProgram, ShaderError, emit, io, range, resource};

const HEADER: &str = "victory-shader-fingerprints-v1";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Emitted(Hash, Hash),
    Refused(Hash),
}

/// Hash every host binding, including exact sampler float bits. The numeric
/// encoding is local to this manifest version and independent of paths or OS.
fn bindings(shader: &HostShader) -> Hash {
    use victory_shader::{HwStage, RootType};
    let HostShader {
        stage,
        dxil: _,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        memory,
        memory_writes,
        view_addresses,
        heap,
        heap_writes,
        view_marks,
    } = shader;
    let mut h = blake3::Hasher::new();
    let mut word = |x: u32| {
        h.update(&x.to_le_bytes());
    };
    word(match stage {
        HwStage::Ls => 0,
        HwStage::Hs => 1,
        HwStage::Es => 2,
        HwStage::Gs => 3,
        HwStage::Vs => 4,
        HwStage::Ps => 5,
        HwStage::Ms => 6,
        HwStage::Cs => 7,
    });
    word(user_data.len() as u32);
    for u in user_data {
        word(u.index);
        word(u32::from(u.slot));
        let (tag, value) = match u.root {
            RootType::Constants => (0, 0),
            RootType::Descriptor => (1, 0),
            RootType::Table => (2, 0),
            RootType::ImmediateSampler => (3, 0),
            RootType::ResourceHeap => (4, 0),
            RootType::SamplerHeap => (5, 0),
            RootType::Other(n) => (6, n),
        };
        word(tag);
        word(value);
        word(u.dword);
    }
    word(samplers.len() as u32);
    for sampler in samplers {
        for &v in sampler {
            word(v);
        }
    }
    word(static_samplers.len() as u32);
    for s in static_samplers {
        for v in [
            s.filter,
            s.address[0],
            s.address[1],
            s.address[2],
            s.mip_lod_bias.to_bits(),
            s.max_anisotropy,
            s.comparison,
            s.border,
            s.min_lod.to_bits(),
            s.max_lod.to_bits(),
        ] {
            word(v);
        }
    }
    for slots in [root_cbvs, root_srvs, root_uavs] {
        word(slots.len() as u32);
        for &v in slots {
            word(v.into());
        }
    }
    for v in [
        memory,
        memory_writes,
        view_addresses,
        heap,
        heap_writes,
        view_marks,
    ] {
        word(u32::from(*v));
    }
    h.finalize()
}

fn program(container: &Container, native: &NativeProgram) -> Result<HostShader, ShaderError> {
    // Same stages and write marks as the runtime translator, for each native
    // program in the input rather than just the first supported stage.
    let (ir, structured) = range::build(&native.patched_code(), native.wave64, native.vgprs)?;
    let resources = resource::track(&ir, &native.layout, native.stage, structured.as_ref().ok())?;
    let io = io::raise(&ir, native)?;
    emit::emit_with(
        container,
        native,
        &ir,
        &structured?,
        &io,
        &resources,
        Options { view_marks: true },
    )
}

#[derive(Default)]
pub(crate) struct Batch {
    rows: BTreeMap<String, Outcome>,
    unreadable: usize,
}

pub(crate) fn collect(
    source: &crate::shader::Source,
    seen: &Mutex<HashSet<Hash>>,
    batch: &Mutex<Batch>,
) {
    let Some(bytes) = source.bytes() else {
        batch.lock().unwrap().unreadable += 1;
        return;
    };
    let container = Container::parse(&bytes);
    // A raw source includes subsequent file data. The DXBC parser validates
    // the declared size before we use it as the identity of this input.
    let input = match &container {
        Ok(_) => &bytes[..u32::from_le_bytes(bytes[24..28].try_into().unwrap()) as usize],
        Err(_) => &bytes[..],
    };
    let hash = blake3::hash(input);
    if !seen.lock().unwrap().insert(hash) {
        return;
    }
    let refusal = |error: ShaderError| Outcome::Refused(blake3::hash(error.to_string().as_bytes()));
    let mut results = Vec::new();
    let parsed = container.and_then(|c| {
        let part = c
            .part(FourCC::DXIL)
            .or(c.part(FourCC::ACCC))
            .ok_or(ShaderError::Malformed("no DXIL part"))?;
        let programs = victory_shader::native_programs(part.data)?;
        if programs.is_empty() {
            return Err(ShaderError::Malformed("no native programs"));
        }
        for (index, native) in programs.iter().enumerate() {
            let outcome = match program(&c, native) {
                Ok(s) => Outcome::Emitted(blake3::hash(&s.dxil), bindings(&s)),
                Err(e) => refusal(e),
            };
            results.push((format!("{hash}/{index}"), outcome));
        }
        Ok(())
    });
    if let Err(e) = parsed {
        results.push((format!("{hash}/container"), refusal(e)));
    }
    batch.lock().unwrap().rows.extend(results);
}

impl Batch {
    pub(crate) fn save(&self, path: &Path) -> Result<()> {
        ensure!(
            self.unreadable == 0,
            "{} packed records could not be unpacked; comparison incomplete",
            self.unreadable
        );
        ensure!(!self.rows.is_empty(), "no shader inputs found");
        let mut data = format!("{HEADER} {}\n", emit::loop_limit());
        for (key, value) in &self.rows {
            match value {
                Outcome::Emitted(dxil, bindings) => writeln!(data, "{key} ok {dxil} {bindings}")?,
                Outcome::Refused(reason) => writeln!(data, "{key} refused {reason}")?,
            }
        }
        writeln!(data, "end {}", self.rows.len())?;
        let dir = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let mut file = tempfile::NamedTempFile::new_in(dir)?;
        file.write_all(data.as_bytes())?;
        file.persist_noclobber(path)
            .context("could not save new fingerprint file")?;
        println!("fingerprints written: {}", self.rows.len());
        Ok(())
    }
}

fn parse(text: &str) -> Result<(u32, BTreeMap<String, Outcome>)> {
    let mut lines = text.lines();
    let limit = lines
        .next()
        .and_then(|line| line.strip_prefix(HEADER))
        .and_then(|rest| rest.strip_prefix(' '))
        .and_then(|s| s.parse::<u32>().ok())
        .context("unsupported fingerprint header")?;
    let mut rows = BTreeMap::new();
    let count = lines
        .next_back()
        .and_then(|line| line.strip_prefix("end "))
        .and_then(|n| n.parse::<usize>().ok())
        .context("incomplete fingerprint file")?;
    for line in lines {
        let fields: Vec<_> = line.split_ascii_whitespace().collect();
        let (key, value) = match fields.as_slice() {
            [key, "ok", d, b] => (
                *key,
                Outcome::Emitted(Hash::from_hex(d)?, Hash::from_hex(b)?),
            ),
            [key, "refused", r] => (*key, Outcome::Refused(Hash::from_hex(r)?)),
            _ => anyhow::bail!("invalid fingerprint row"),
        };
        let (id, program) = key.split_once('/').context("missing program index")?;
        ensure!(
            Hash::from_hex(id)?.to_hex().as_str() == id
                && (program == "container"
                    || program
                        .parse::<usize>()
                        .ok()
                        .is_some_and(|n| n.to_string() == program)),
            "invalid input key"
        );
        ensure!(
            rows.insert(key.to_owned(), value).is_none(),
            "duplicate input key"
        );
    }
    ensure!(
        count > 0 && count == rows.len(),
        "fingerprint count differs from footer"
    );
    Ok((limit, rows))
}

pub(crate) fn compare(before: &Path, after: &Path) -> Result<()> {
    let (a, before) = parse(&std::fs::read_to_string(before)?)?;
    let (b, after) = parse(&std::fs::read_to_string(after)?)?;
    ensure!(a == b, "different shader loop limits");
    #[derive(Default)]
    struct Counts {
        before_only: usize,
        identical: usize,
        dxil: usize,
        bindings: usize,
        both: usize,
        now_refused: usize,
        now_emitted: usize,
        same_refusal: usize,
        different_refusal: usize,
    }
    let mut counts = Counts::default();
    for (key, old) in &before {
        let count = match (old, after.get(key)) {
            (_, None) => &mut counts.before_only,
            (Outcome::Emitted(dxil, bindings), Some(Outcome::Emitted(new_dxil, new_bindings))) => {
                match (dxil == new_dxil, bindings == new_bindings) {
                    (true, true) => &mut counts.identical,
                    (false, true) => &mut counts.dxil,
                    (true, false) => &mut counts.bindings,
                    (false, false) => &mut counts.both,
                }
            }
            (Outcome::Emitted(..), Some(Outcome::Refused(_))) => &mut counts.now_refused,
            (Outcome::Refused(_), Some(Outcome::Emitted(..))) => &mut counts.now_emitted,
            (Outcome::Refused(a), Some(Outcome::Refused(b))) if a == b => &mut counts.same_refusal,
            (Outcome::Refused(_), Some(Outcome::Refused(_))) => &mut counts.different_refusal,
        };
        *count += 1;
    }
    let after_only = after.keys().filter(|k| !before.contains_key(*k)).count();
    for (label, n) in [
        ("before only", counts.before_only),
        ("after only", after_only),
        ("identical emitted output", counts.identical),
        ("DXIL only changed", counts.dxil),
        ("bindings only changed", counts.bindings),
        ("DXIL and bindings changed", counts.both),
        ("emitted to refused", counts.now_refused),
        ("refused to emitted", counts.now_emitted),
        ("identical refusal", counts.same_refusal),
        ("refusal reason changed", counts.different_refusal),
    ] {
        println!("{label}: {n}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_fingerprints_are_independent_of_dxil() {
        let mut shader = emit::host::stub_ps();
        let original = bindings(&shader);
        shader.dxil.push(0x55);
        assert_eq!(bindings(&shader), original);
        shader.root_srvs.push(3);
        assert_ne!(bindings(&shader), original);
        shader.root_srvs.pop();
        shader.view_marks = !shader.view_marks;
        assert_ne!(bindings(&shader), original);
    }

    #[test]
    fn manifests_match_by_input_and_reject_duplicate_or_partial_rows() {
        let a = blake3::hash(b"a");
        let b = blake3::hash(b"b");
        let row_a = format!("{a}/0 ok {b} {a}");
        let row_b = format!("{a}/1 refused {b}");
        let header = format!("{HEADER} 16");
        let forward = format!("{header}\n{row_a}\n{row_b}\nend 2\n");
        let reverse = format!("{header}\n{row_b}\n{row_a}\nend 2\n");
        assert!(parse(&forward).unwrap() == parse(&reverse).unwrap());
        assert!(parse(&format!("{header}\n{row_a}\n{row_a}\nend 2\n")).is_err());
        assert!(parse(&format!("{header}\n{row_a}\n")).is_err());
    }
}
