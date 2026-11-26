//! Textures the title reads in a format of another family than the one the
//! host texture has (decision 0007, "One host texture per texture memory"):
//! Grand Theft Auto VI's compute writes two 192x108 lighting textures as
//! R32_UINT and its lighting reads the same memory as R9G9B9E5_SHAREDEXP,
//! a view the host does not allow of an R32 texture (refused: the lighting
//! read zeros and the scene stayed black). A shader-resource view in a
//! format of the texture's texel size gets a twin texture in that format,
//! whose bytes are copied from the texture before every submission (through
//! a buffer: placed footprints copy between formats of one texel size).
//!
//! shortcut: every twin is copied at every submission, changed or not, and
//! only shader-resource views use twins (none writes them back). Upgrade
//! when the copies (counted on the frame-rate line) cost, or a title
//! writes such a view.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, Weak};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::host::{host, transition};
use crate::resource::HostTexture;

const COPY_DEST: u32 = 0x400;
const COPY_SOURCE: u32 = 0x800;
/// Pixel and non-pixel shader resource.
const SHADER_RESOURCE: u32 = 0xc0;

/// Copies made into twins since the last frame-rate line.
pub(crate) static COPIES: AtomicU64 = AtomicU64::new(0);

struct FormatTwin {
    source: Weak<HostTexture>,
    format: u32,
    twin: Arc<HostTexture>,
    /// Per subresource: its place in the scratch buffer, as the source
    /// and as the twin have it (the same bytes).
    routes: Vec<(u32, D3D12_PLACED_SUBRESOURCE_FOOTPRINT, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)>,
    scratch: ID3D12Resource,
}

// SAFETY: D3D12 objects are free-threaded; the rest is data.
unsafe impl Send for FormatTwin {}
// SAFETY: as above.
unsafe impl Sync for FormatTwin {}

static TWINS: Mutex<Vec<Arc<FormatTwin>>> = Mutex::new(Vec::new());

/// Whether a shader-resource view of `texture` (a host texture of format
/// `own`) in format `view` needs a twin: another family of the same texel
/// size, neither depth nor block-compressed.
pub(crate) fn wanted(own: u32, view: u32) -> bool {
    use crate::resource::{DEPTH_FORMATS, element, family};
    if view == 0 || family(view) == family(own) {
        return false;
    }
    let depth = |f: u32| DEPTH_FORMATS.iter().any(|&(typed, _)| typed == f) || (19..=22).contains(&f) || (39..=41).contains(&f);
    if depth(own) || depth(view) {
        return false;
    }
    matches!((element(own), element(view)), (Some((a, 1)), Some((b, 1))) if a == b)
}

/// The twin of `source` in `format`, made on first use. Its contents are
/// the source's as of the last submission.
pub(crate) fn twin(source: &Arc<HostTexture>, format: u32) -> Option<Arc<HostTexture>> {
    let mut twins = TWINS.lock().unwrap_or_else(|e| e.into_inner());
    twins.retain(|t| t.source.strong_count() > 0);
    if let Some(t) = twins
        .iter()
        .find(|t| t.format == format && std::ptr::eq(t.source.as_ptr(), Arc::as_ptr(source)))
    {
        return Some(t.twin.clone());
    }
    let made = make(source, format)?;
    let texture = made.twin.clone();
    twins.push(Arc::new(made));
    Some(texture)
}

fn make(source: &Arc<HostTexture>, format: u32) -> Option<FormatTwin> {
    let host = host()?;
    // SAFETY: a live resource.
    let sd = unsafe { source.texture.GetDesc() };
    if sd.SampleDesc.Count > 1 {
        return None;
    }
    let desc = D3D12_RESOURCE_DESC {
        Format: DXGI_FORMAT(format as i32),
        Flags: D3D12_RESOURCE_FLAG_NONE,
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Alignment: 0,
        ..sd
    };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; our out pointer.
    let created = unsafe {
        host.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut texture)
    };
    if let Err(e) = created {
        tracing::warn!(error = %e, format, own = sd.Format.0, "no twin for a view in a format of another family");
        return None;
    }
    let texture = texture?;
    crate::host::name(&texture, || format!("format twin {}x{} f{format}", sd.Width, sd.Height));
    let subs = if sd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(sd.MipLevels)
    } else {
        u32::from(sd.MipLevels) * u32::from(sd.DepthOrArraySize).max(1)
    };
    let footprints = |d: &D3D12_RESOURCE_DESC| {
        let mut f = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); subs as usize];
        let mut rows = vec![0u32; subs as usize];
        // SAFETY: outputs sized for `subs` subresources.
        unsafe { host.device.GetCopyableFootprints(d, 0, subs, 0, Some(f.as_mut_ptr()), Some(rows.as_mut_ptr()), None, None) };
        (f, rows)
    };
    let (sf, rows) = footprints(&sd);
    let (tf, _) = footprints(&desc);
    let mut at = 0u64;
    let mut routes = Vec::new();
    for s in 0..subs as usize {
        let (mut a, mut b) = (sf[s], tf[s]);
        if a.Footprint.RowPitch != b.Footprint.RowPitch || a.Footprint.Width != b.Footprint.Width {
            tracing::warn!(format, own = sd.Format.0, "a format twin whose layout differs from its texture's: not made");
            return None;
        }
        (a.Offset, b.Offset) = (at, at);
        let bytes = u64::from(a.Footprint.RowPitch) * u64::from(rows[s]) * u64::from(a.Footprint.Depth);
        at = (at + bytes).next_multiple_of(512);
        routes.push((s as u32, a, b));
    }
    let scratch = crate::twin::shared_scratch(at)?;
    tracing::info!(format, own = sd.Format.0, width = sd.Width, height = sd.Height, "a view in a format of another family reads a twin of its texture");
    Some(FormatTwin {
        source: Arc::downgrade(source),
        format,
        twin: Arc::new(HostTexture::new(texture, 0, Vec::new())),
        routes,
        scratch,
    })
}

/// Whether any twin is to be copied at submission.
pub(crate) fn any() -> bool {
    !TWINS.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
}

/// Records into a submission's prelude `l` the copy of every twin's
/// source into it, from the source's real state and back; the twins are
/// left readable by shaders.
pub(crate) fn refresh(l: &ID3D12GraphicsCommandList) {
    let twins: Vec<Arc<FormatTwin>> = TWINS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for t in twins {
        let Some(source) = t.source.upgrade() else { continue };
        let real = source.state.load(Relaxed);
        let wanted = crate::host::legal_state(&source.texture, COPY_SOURCE);
        let twin_real = t.twin.state.load(Relaxed);
        let at = |r: &ID3D12Resource, sub: u32| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the copy does not release.
            pResource: unsafe { std::mem::transmute_copy(r) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub },
        };
        let placed = |f: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&t.scratch) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
        };
        let st = |r: &ID3D12Resource, a: u32, b: u32| transition(r, D3D12_RESOURCE_STATES(a as i32), D3D12_RESOURCE_STATES(b as i32));
        // SAFETY: an open list; live resources in the states named (the
        // scratch buffer is common between copies: its first copy promotes
        // it, as for the depth twins sharing it).
        unsafe {
            if real != wanted {
                l.ResourceBarrier(&[st(&source.texture, real, wanted)]);
            }
            for (sub, from, _) in &t.routes {
                l.CopyTextureRegion(&placed(*from), 0, 0, 0, &at(&source.texture, *sub), None);
            }
            let mut b = vec![st(&t.scratch, COPY_DEST, COPY_SOURCE)];
            if twin_real != COPY_DEST {
                b.push(st(&t.twin.texture, twin_real, COPY_DEST));
            }
            l.ResourceBarrier(&b);
            for (sub, _, to) in &t.routes {
                l.CopyTextureRegion(&at(&t.twin.texture, *sub), 0, 0, 0, &placed(*to), None);
            }
            let mut b = vec![st(&t.scratch, COPY_SOURCE, 0), st(&t.twin.texture, COPY_DEST, SHADER_RESOURCE)];
            if real != wanted {
                b.push(st(&source.texture, wanted, real));
            }
            l.ResourceBarrier(&b);
        }
        t.twin.state.store(SHADER_RESOURCE, Relaxed);
        COPIES.fetch_add(1, Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Grand Theft Auto VI's lighting textures: written as R32_UINT, read
    /// as R9G9B9E5. The twin holds the texture's very bytes once a
    /// submission's prelude ran.
    #[test]
    fn a_twin_takes_its_textures_bytes_at_submission() {
        check_twin_write_order(false);
    }

    /// Compare a write and a later read in one command list with a read
    /// submitted after the write. A same-submission twin should see each write.
    #[test]
    #[ignore = "the fallback twin is currently refreshed only at submission start"]
    fn a_twin_sees_an_earlier_write_in_the_same_list() {
        check_twin_write_order(true);
    }

    fn check_twin_write_order(update_in_list: bool) {
        let Some(host) = host() else {
            eprintln!("no host GPU; skipped format-twin readbacks");
            return;
        };
        const W: u32 = 8;
        const H: u32 = 4;
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(W),
            Height: H,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R32_UINT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: plain creation; our out pointer.
        unsafe { host.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut texture) }.unwrap();
        let source = Arc::new(HostTexture::new(texture.unwrap(), COPY_DEST, Vec::new()));
        assert!(wanted(42, 67), "R9G9B9E5 of R32_UINT");
        assert!(!wanted(42, 41), "R32_FLOAT is of the texture's family");
        let twin = twin(&source, 67).expect("a twin");
        // SAFETY: a live resource.
        assert_eq!(unsafe { twin.texture.GetDesc() }.Format.0, 67);

        const PITCH: u32 = 256;
        let texel = |x: u32, y: u32| 0x1234_5000 ^ (y << 8) ^ x;
        let footprint = |format: DXGI_FORMAT| D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: format, Width: W, Height: H, Depth: 1, RowPitch: PITCH },
        };
        let at = |r: &ID3D12Resource| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the copy does not release.
            pResource: unsafe { std::mem::transmute_copy(r) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        let placed = |r: &ID3D12Resource, f| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(r) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
        };
        let write_source = |list: &crate::host::HostList, mask: u32| {
            let (buffer, data) = list.upload(host, u64::from(PITCH * H)).unwrap();
            for y in 0..H {
                for x in 0..W {
                    // SAFETY: this texel lies in the mapped upload buffer.
                    unsafe {
                        data.add((y * PITCH + x * 4) as usize)
                            .cast::<u32>()
                            .write_unaligned(texel(x, y) ^ mask)
                    };
                }
            }
            // SAFETY: open list, source texture in COPY_DEST.
            unsafe {
                list.list.CopyTextureRegion(
                    &at(&source.texture),
                    0,
                    0,
                    0,
                    &placed(&buffer, footprint(DXGI_FORMAT_R32_UINT)),
                    None,
                )
            };
        };
        let seed = host.new_list().unwrap();
        write_source(&seed, 0);
        seed.close();
        host.execute(&[&seed]);

        let check = |write: bool, mask: u32| {
            let list = host.new_list().unwrap();
            if write {
                write_source(&list, mask);
            }
            let copy_out = |texture: &ID3D12Resource, format, state| {
                let (buffer, bytes) = list.readback(host, u64::from(PITCH * H)).unwrap();
                let barrier = |a: u32, b: u32| {
                    transition(
                        texture,
                        D3D12_RESOURCE_STATES(a as i32),
                        D3D12_RESOURCE_STATES(b as i32),
                    )
                };
                // SAFETY: each texture is transitioned for the copy and restored before returning.
                unsafe {
                    list.list.ResourceBarrier(&[barrier(state, COPY_SOURCE)]);
                    list.list.CopyTextureRegion(
                        &placed(&buffer, footprint(format)),
                        0,
                        0,
                        0,
                        &at(texture),
                        None,
                    );
                    list.list.ResourceBarrier(&[barrier(COPY_SOURCE, state)]);
                }
                (buffer, bytes)
            };
            let (_original, original_bytes) =
                copy_out(&source.texture, DXGI_FORMAT_R32_UINT, COPY_DEST);
            let (_mirror, mirror_bytes) = copy_out(&twin.texture, DXGI_FORMAT(67), SHADER_RESOURCE);
            list.close();
            host.execute(&[&list]);
            let mut matching = 0;
            let mut previous = 0;
            for y in 0..H {
                for x in 0..W {
                    let offset = (y * PITCH + x * 4) as usize;
                    // SAFETY: both mapped ranges contain these texels and the submitted work has finished.
                    let (original, mirror) = unsafe {
                        (
                            original_bytes.add(offset).cast::<u32>().read_unaligned(),
                            mirror_bytes.add(offset).cast::<u32>().read_unaligned(),
                        )
                    };
                    assert_eq!(original, texel(x, y) ^ mask, "source {x},{y}");
                    matching += u32::from(mirror == original);
                    previous += u32::from(mirror == texel(x, y));
                }
            }
            eprintln!(
                "twin readback: same-list write={write}, source={}/{}, twin={matching}/{}, seed={previous}/{}",
                W * H,
                W * H,
                W * H,
                W * H
            );
            matching
        };
        let mask = if update_in_list { u32::MAX } else { 0 };
        let first = check(update_in_list, mask);
        if update_in_list {
            assert_eq!(check(false, mask), W * H, "next submission");
        }
        assert_eq!(first, W * H, "twin after source write");
        assert_eq!(
            source.state.load(Relaxed),
            COPY_DEST,
            "the texture left as it was"
        );
    }
}
