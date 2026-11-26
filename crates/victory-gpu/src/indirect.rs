//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per compute dispatch, 65,535 groups per
//! mesh dispatch, the budget) cut legitimate work beyond them; raise them
//! if the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, PCSTR, s};

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), SRV(t0), UAV(u1), UAV(u0)"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint unused; };
// The title's records, read only; the clamped ones go to the list's own.
ByteAddressBuffer args : register(t0);
RWByteAddressBuffer dest : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
// Clamps the record at `at` into `dest`; its cost.
uint clamped(uint at) {
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(args.Load(at), 1u << 24);
        uint instances = min(args.Load(at + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        dest.Store2(at, uint2(count, instances));
        // The rest of the record as it is: first vertex and instance (4 words
        // in all), or first index, base vertex and first instance (5).
        if (kind == 1) {
            dest.Store3(at + 8, args.Load3(at + 8));
        } else {
            dest.Store2(at + 8, args.Load2(at + 8));
        }
    } else if (kind == 2) {
        uint3 g = min(args.Load3(at), uint3(65535, 65535, 65535));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        dest.Store3(at, g);
    } else {
        // The translated mesh group index reads GroupID.x. Dispatch the
        // full in-budget count along X so each group gets its own index.
        static const uint MESH_GROUPS = 65535u;
        uint3 g = args.Load3(at);
        // Within the axis limits x * y fits in 32 bits. Divide the
        // ceiling by z to check the total without overflowing.
        if (any(g > MESH_GROUPS)
                || (g.z != 0 && g.x * g.y > MESH_GROUPS / max(g.z, 1u))) {
            g = min(g, uint3(MESH_GROUPS, 1u, 1u));
        } else {
            g = uint3(g.x * g.y * g.z, 1u, 1u);
        }
        cost = g.x * g.y * g.z * 128;
        dest.Store3(at, g);
    }
    return cost;
}
// A record past the budget (no cost reaches it).
static const uint PAST = 0xffffffffu;
groupshared uint costs[64];
// One group walks the records in tiles of 64: every lane clamps its own
// record, then the first lane charges the tile's records in order. No
// other group shares the budget: a shared add wraps once enough records
// are rejected, and a compare-exchange retries more the more records fit.
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint left = 0;
    if (lane == 0) {
        left = BUDGET - min(budget.Load(0), BUDGET);
    }
    for (uint tile = 0; tile < records / 64 + min(records % 64, 1u); tile++) {
        uint record = tile * 64 + lane;
        uint at = record * stride;
        uint cost = 0;
        if (record < records) {
            cost = clamped(at);
        }
        costs[lane] = cost;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0) {
            for (uint i = 0; i < 64; i++) {
                if (costs[i] > left) {
                    // The first record that does not fit takes what is
                    // left, so those after it run empty too.
                    costs[i] = PAST;
                    left = 0;
                } else {
                    left -= costs[i];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (costs[lane] == PAST) {
            // Draws keep their layout with no vertices; groups become zero.
            dest.Store(at, 0);
        }
    }
    if (lane == 0) {
        budget.Store(0, BUDGET - left);
    }
}
"#;

const BATCH_SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
cbuffer Batch : register(b0) { uint first_job; uint job_count; uint unused0; uint unused1; };
// Packed 32-byte job descriptors:
//   j0 = uint4(kind, stride, records, src_offset)
//   j1 = uint4(dst_offset, 0, 0, 0)
ByteAddressBuffer jobs : register(t0);
ByteAddressBuffer src_args : register(t1);
RWByteAddressBuffer dst_args : register(u0);
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
static const uint GROUPS = 1u << 20;
static const uint PAST = 0xffffffffu;
groupshared uint costs[64];
groupshared uint single_run;

uint compute_clamped(uint kind, uint src_at, out uint2 draw_counts, out uint3 groups) {
    draw_counts = uint2(0, 0);
    groups = uint3(0, 0, 0);
    [branch]
    if (kind <= 1) {
        uint count = min(src_args.Load(src_at), 1u << 24);
        uint instances = min(src_args.Load(src_at + 4), (1u << 26) / max(count, 1u));
        draw_counts = uint2(count, instances);
        return count * instances;
    } else if (kind == 2) {
        uint3 g = min(src_args.Load3(src_at), uint3(65535u, 65535u, 65535u));
        if (g.z != 0) {
            if (g.x * g.y > GROUPS / max(g.z, 1u)) {
                g = uint3(0, 0, 0);
            }
        }
        groups = g;
        return g.x * g.y * g.z * 64u;
    } else {
        static const uint MESH_GROUPS = 65535u;
        uint3 g = src_args.Load3(src_at);
        if (any(g > MESH_GROUPS)
                || (g.z != 0 && g.x * g.y > MESH_GROUPS / max(g.z, 1u))) {
            g = min(g, uint3(MESH_GROUPS, 1u, 1u));
        } else {
            // A translated mesh reads only GroupID.x, as in the inline clamp.
            g = uint3(g.x * g.y * g.z, 1u, 1u);
        }
        groups = g;
        return g.x * g.y * g.z * 128u;
    }
}

void write_record(uint kind, uint stride, uint src_at, uint dst_at, bool over, uint2 draw_counts, uint3 groups) {
    if (kind <= 1) {
        if (over) {
            draw_counts.x = 0;
        }
        dst_args.Store2(dst_at, draw_counts);
        for (uint off = 8; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    } else {
        if (over) {
            groups.x = 0;
        }
        dst_args.Store3(dst_at, groups);
        for (uint off = 12; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    }
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint left = 0;
    if (lane == 0) {
        left = BUDGET - min(budget.Load(0), BUDGET);
    }
    uint j = first_job;
    uint end = first_job + job_count;
    while (j < end) {
        // Fast path: clamp up to 64 consecutive single-record jobs in one wave.
        // Guard root-SRV loads with `[branch]` so inactive tail lanes never
        // load past the end of `jobs` (root descriptors have no HW bounds check).
        uint4 j0 = uint4(0, 0, 0, 0);
        uint4 j1 = uint4(0, 0, 0, 0);
        [branch]
        if (j + lane < end) {
            j0 = jobs.Load4((j + lane) * 32);
            j1 = jobs.Load4((j + lane) * 32 + 16);
        }
        costs[lane] = j0.z;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0) {
            uint run = 0;
            // Avoid `&&` here: FXC evaluates both sides, which would read
            // `costs[64]` past the 64-element groupshared array.
            [loop]
            while (run < 64) {
                if (costs[run] != 1) {
                    break;
                }
                run++;
            }
            single_run = run;
        }
        GroupMemoryBarrierWithGroupSync();
        uint run = single_run;
        if (run > 0) {
            uint cost = 0;
            uint2 draw_counts = uint2(0, 0);
            uint3 groups = uint3(0, 0, 0);
            [branch]
            if (lane < run) {
                cost = compute_clamped(j0.x, j0.w, draw_counts, groups);
            }
            costs[lane] = cost;
            GroupMemoryBarrierWithGroupSync();
            if (lane == 0) {
                for (uint i = 0; i < run; i++) {
                    if (costs[i] > left) {
                        costs[i] = PAST;
                        left = 0;
                    } else {
                        left -= costs[i];
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
            [branch]
            if (lane < run) {
                write_record(j0.x, j0.y, j0.w, j1.x, costs[lane] == PAST, draw_counts, groups);
            }
            j += run;
        } else {
            uint4 m0 = jobs.Load4(j * 32);
            uint4 m1 = jobs.Load4(j * 32 + 16);
            uint tiles = m0.z / 64 + min(m0.z % 64, 1u);
            for (uint tile = 0; tile < tiles; tile++) {
                uint record = tile * 64 + lane;
                uint src_at = m0.w + record * m0.y;
                uint dst_at = m1.x + record * m0.y;
                uint cost = 0;
                uint2 draw_counts = uint2(0, 0);
                uint3 groups = uint3(0, 0, 0);
                [branch]
                if (record < m0.z) {
                    cost = compute_clamped(m0.x, src_at, draw_counts, groups);
                }
                costs[lane] = cost;
                GroupMemoryBarrierWithGroupSync();
                if (lane == 0) {
                    for (uint i = 0; i < 64; i++) {
                        if (costs[i] > left) {
                            costs[i] = PAST;
                            left = 0;
                        } else {
                            left -= costs[i];
                        }
                    }
                }
                GroupMemoryBarrierWithGroupSync();
                [branch]
                if (record < m0.z) {
                    write_record(m0.x, m0.y, src_at, dst_at, costs[lane] == PAST, draw_counts, groups);
                }
            }
            j += 1;
        }
    }
    if (lane == 0) {
        budget.Store(0, BUDGET - left);
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match compile(SOURCE, s!("clamp")) {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn batch_clamp() -> Option<&'static Clamp> {
    static BATCH: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    BATCH
        .get_or_init(|| match compile(BATCH_SOURCE, s!("batch_clamp")) {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no batched argument clamp: falling back to inline");
                None
            }
        })
        .as_ref()
}

fn compile(source: &str, name: PCSTR) -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            name,
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

const HOIST_CHUNK: u64 = 256 << 10;

struct Job {
    src_gpu: u64,
    dst_gpu: u64,
}

struct JobUpload {
    _buffer: ID3D12Resource,
    mapped: usize,
    gpu: u64,
    capacity: usize,
}

// SAFETY: `mapped` is a persistent upload mapping written only under the
// list's `HoistedClamps` mutex while the list is open for recording.
unsafe impl Send for JobUpload {}

/// Indirect argument clamps hoisted from a [`HostList`]'s initial non-writing
/// prefix into its submission prelude. Each hoisted draw reads the title's
/// argument buffer through an SRV (never transitioning or mutating it) and
/// writes clamped records into the list's own scratch chunks, which move
/// `COMMON -> UNORDERED_ACCESS -> INDIRECT_ARGUMENT` in the prelude and back
/// to `COMMON` once at [`Self::end`] (or [`HostList::close`]).
pub(crate) struct HoistedClamps {
    hoistable: bool,
    restored: bool,
    jobs: Vec<Job>,
    payload: Vec<[u32; 8]>,
    upload: Option<JobUpload>,
    chunks: Vec<ID3D12Resource>,
    used: usize,
    at: u64,
    kept: Vec<ID3D12Resource>,
}

// SAFETY: D3D12 resources are free-threaded; mutated under the list's mutex.
unsafe impl Send for HoistedClamps {}

impl Default for HoistedClamps {
    fn default() -> Self {
        Self {
            hoistable: true,
            restored: false,
            jobs: Vec::new(),
            payload: Vec::new(),
            upload: None,
            chunks: Vec::new(),
            used: 0,
            at: 0,
            kept: Vec::new(),
        }
    }
}

impl HoistedClamps {
    pub(crate) fn reset(&mut self) {
        self.hoistable = true;
        self.restored = false;
        self.jobs.clear();
        self.payload.clear();
        self.used = 0;
        self.at = 0;
        self.kept.clear();
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    fn min_stride(kind: u32) -> Option<u32> {
        match kind {
            0 => Some(16),
            1 => Some(20),
            2 | 10 => Some(12),
            _ => None,
        }
    }

    fn byte_len(gpu: u64, (kind, stride, records): (u32, u32, u32)) -> Option<u64> {
        let min = Self::min_stride(kind)?;
        if records == 0 || stride < min || !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
            return None;
        }
        let len = u64::from(stride).checked_mul(u64::from(records))?;
        let aligned = len.checked_next_multiple_of(16)?;
        (aligned <= HOIST_CHUNK).then_some(len)
    }

    pub(crate) fn can_accept(&self, gpu: u64, spec: (u32, u32, u32)) -> bool {
        self.hoistable
            && u32::try_from(self.jobs.len()).is_ok()
            && batch_clamp().is_some()
            && Self::byte_len(gpu, spec).is_some()
    }

    fn keep_once(&mut self, resource: &ID3D12Resource) {
        let raw = resource.as_raw();
        if !self.kept.iter().any(|r| r.as_raw() == raw) {
            self.kept.push(resource.clone());
        }
    }

    fn alloc_room(&mut self, bytes: u64) -> Option<(ID3D12Resource, u64)> {
        let aligned = bytes.next_multiple_of(16);
        if aligned > HOIST_CHUNK {
            return None;
        }
        if self.used > 0 && self.at + aligned <= HOIST_CHUNK {
            let at = self.at;
            self.at += aligned;
            return Some((self.chunks[self.used - 1].clone(), at));
        }
        if self.used == self.chunks.len() {
            let (buffer, _) = host()?.shared_buffer(HOIST_CHUNK)?;
            crate::host::name(&buffer, || "hoisted indirect arguments".into());
            self.chunks.push(buffer);
        }
        let i = self.used;
        self.used += 1;
        self.at = aligned;
        Some((self.chunks[i].clone(), 0))
    }

    fn ensure_upload(&mut self, needed_jobs: usize) -> bool {
        if self.upload.as_ref().is_some_and(|u| u.capacity >= needed_jobs) {
            return true;
        }
        let Some(host) = host() else {
            return false;
        };
        let capacity = needed_jobs.next_power_of_two().max(256);
        let Some(bytes) = (capacity as u64).checked_mul(32) else {
            return false;
        };
        let Some((buffer, data)) = host.upload_buffer(bytes) else {
            return false;
        };
        crate::host::name(&buffer, || format!("hoisted clamp jobs ({capacity})"));
        // SAFETY: a live upload buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if self.upload.is_some() && !self.payload.is_empty() {
            // SAFETY: `data` has room for `capacity >= self.payload.len()` 32-byte jobs.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.payload.as_ptr().cast::<u8>(),
                    data,
                    self.payload.len() * 32,
                );
            }
        }
        self.upload = Some(JobUpload {
            _buffer: buffer,
            mapped: data as usize,
            gpu,
            capacity,
        });
        true
    }

    pub(crate) fn push(
        &mut self,
        src_buf: &ID3D12Resource,
        gpu: u64,
        spec: (u32, u32, u32),
    ) -> Option<(ID3D12Resource, u64)> {
        if !self.can_accept(gpu, spec) {
            return None;
        }
        let len = Self::byte_len(gpu, spec)?;
        let slot = self.jobs.len();
        if !self.ensure_upload(slot + 1) {
            return None;
        }
        let (dst, dst_at) = self.alloc_room(len)?;
        // SAFETY: live D3D12 buffers.
        let (buf_gpu, dst_gpu) = unsafe { (src_buf.GetGPUVirtualAddress(), dst.GetGPUVirtualAddress()) };
        let (src_gpu, src_offset) = match gpu.checked_sub(buf_gpu) {
            Some(rel) if rel.checked_add(len).is_some_and(|e| e <= u64::from(u32::MAX)) => (buf_gpu, rel as u32),
            _ => (gpu, 0),
        };
        self.keep_once(src_buf);
        let (kind, stride, records) = spec;
        let words = [kind, stride, records, src_offset, dst_at as u32, 0, 0, 0];
        if let Some(u) = &self.upload {
            // SAFETY: `ensure_upload(slot + 1)` guarantees `[slot * 32 .. (slot + 1) * 32)`
            // lies within the persistent mapped upload buffer.
            unsafe {
                (u.mapped as *mut [u32; 8]).add(slot).write(words);
            }
        }
        self.payload.push(words);
        self.jobs.push(Job { src_gpu, dst_gpu });
        Some((dst, dst_at))
    }

    /// Ends the hoistable prefix on `list`: transitions every used scratch
    /// chunk back from `INDIRECT_ARGUMENT` to `COMMON` once, and blocks
    /// further hoisting on this list until `reset()`.
    pub(crate) fn end(&mut self, list: &ID3D12GraphicsCommandList) {
        self.hoistable = false;
        if self.restored || self.used == 0 {
            return;
        }
        self.restored = true;
        let barriers: Vec<D3D12_RESOURCE_BARRIER> = self.chunks[..self.used]
            .iter()
            .map(|b| {
                transition(
                    b,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COMMON,
                )
            })
            .collect();
        // SAFETY: an open list and live scratch chunks in `INDIRECT_ARGUMENT`.
        unsafe { list.ResourceBarrier(&barriers) };
    }

    /// Records the batched hoisted clamp pass(es) into `prelude`, moving each
    /// used scratch chunk `COMMON -> UNORDERED_ACCESS -> INDIRECT_ARGUMENT`.
    pub(crate) fn record_prelude(
        &self,
        prelude: &ID3D12GraphicsCommandList,
        budget_resource: &ID3D12Resource,
        budget_gpu: u64,
    ) -> bool {
        if self.jobs.is_empty() || self.used == 0 {
            return false;
        }
        let (Some(c), Some(upload)) = (batch_clamp(), &self.upload) else {
            return false;
        };
        let mut pre = Vec::with_capacity(self.used + 1);
        // Earlier lists' shader writes to the title's argument buffers first.
        pre.push(uav_barrier_of(None));
        for chunk in &self.chunks[..self.used] {
            pre.push(transition(
                chunk,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ));
        }
        // SAFETY: an open prelude list; root signature, PSO, job buffer,
        // budget buffer, title source buffers (`kept`) and scratch chunks all
        // live through the submission.
        unsafe {
            prelude.ResourceBarrier(&pre);
            prelude.SetComputeRootSignature(&c.root);
            prelude.SetPipelineState(&c.pso);
            prelude.SetComputeRootShaderResourceView(1, upload.gpu);
            prelude.SetComputeRootUnorderedAccessView(4, budget_gpu);
            let mut run_start = 0usize;
            while run_start < self.jobs.len() {
                let key = (self.jobs[run_start].src_gpu, self.jobs[run_start].dst_gpu);
                let mut run_end = run_start + 1;
                while run_end < self.jobs.len()
                    && (self.jobs[run_end].src_gpu, self.jobs[run_end].dst_gpu) == key
                {
                    run_end += 1;
                }
                if run_start > 0 {
                    // Full UAV barrier between runs: keeps both `budget` and
                    // adjacent 16-byte-aligned slices of the same scratch
                    // chunk coherent on wide-cache-line GPUs.
                    prelude.ResourceBarrier(&[uav_barrier_of(None)]);
                }
                let constants = [run_start as u32, (run_end - run_start) as u32, 0, 0];
                prelude.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                prelude.SetComputeRootShaderResourceView(2, key.0);
                prelude.SetComputeRootUnorderedAccessView(3, key.1);
                prelude.Dispatch(1, 1, 1);
                run_start = run_end;
            }
            let mut post = Vec::with_capacity(self.used + 1);
            post.push(uav_barrier_of(Some(budget_resource)));
            for chunk in &self.chunks[..self.used] {
                post.push(transition(
                    chunk,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                ));
            }
            prelude.ResourceBarrier(&post);
        }
        true
    }
}

/// Whether `spec` at `gpu` can be hoisted into `list`'s prelude.
pub(crate) fn can_hoist(list: &HostList, gpu: u64, spec: (u32, u32, u32)) -> bool {
    list.hoisted().can_accept(gpu, spec)
}

/// Queues `spec` at `gpu` in `src_buf` to be clamped in `list`'s prelude into
/// one of `list`'s own scratch chunks. Returns `(chunk, offset)` ready to be
/// read in `INDIRECT_ARGUMENT`, or `None` if hoisting is unavailable.
pub(crate) fn hoist(
    list: &HostList,
    src_buf: &ID3D12Resource,
    gpu: u64,
    spec: (u32, u32, u32),
) -> Option<(ID3D12Resource, u64)> {
    list.budget()?;
    list.hoisted().push(src_buf, gpu, spec)
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` (a title
/// buffer's copy, only read) into the list's own argument buffer, left
/// ready to be read as arguments: that buffer, for [`done`] after the
/// execution. Replaces the list's compute root signature and pipeline
/// (callers bind theirs after). `None` if the clamp is not available.
///
/// Title buffers' copies are never transitioned: they are common between
/// submissions and every use promotes them (shader and root views, the
/// input assembler, these reads), with unordered-access barriers for order.
/// Only the list's argument buffer moves, common, unordered access,
/// indirect argument, common: its states are known whatever order lists
/// run in.
// shortcut: a buffer a shader writes through an unordered view and the
// input assembler then reads in one submission is, by the letter of D3D12,
// promoted to a write state the input assembler may not use (the debug
// layer cannot see view accesses; GPU-based validation might); upgrade to
// per-list tracked buffer states if a driver or that validation minds.
pub(crate) fn bound(list: &HostList, gpu: u64, (kind, stride, records): (u32, u32, u32)) -> Option<ID3D12Resource> {
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return None;
    };
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return None;
    }
    let out = list.arguments(u64::from(stride) * u64::from(records.max(1)))?;
    let l = &list.list;
    let constants = [kind, stride, records, 0];
    // SAFETY: an open list; our root signature, pipeline and buffers, and
    // a title buffer's copy the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.SetComputeRootUnorderedAccessView(3, out.GetGPUVirtualAddress());
        // One group walks every record.
        l.Dispatch(records.min(1), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        ]);
    }
    Some(out)
}

/// After the execution: the list's argument buffer back to common.
pub(crate) fn done(list: &HostList, arguments: &ID3D12Resource) {
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            arguments,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    uav_barrier_of(None)
}

fn uav_barrier_of(resource: Option<&ID3D12Resource>) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(resource.map(|r| {
                    // SAFETY: a borrowed pointer; the barrier never releases it.
                    unsafe { std::mem::transmute_copy(r) }
                })),
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    /// The first word of each clamped record (zero when it runs empty).
    fn clamped(kind: u32, words: &[u32], records: u32, calls: u32) -> Option<Vec<u32>> {
        clamped_words(kind, words, records, calls, 1)
    }

    /// Clamps `records` records of `kind`, filled by repeating `words`,
    /// `calls` times in one list and reads the first `fields` words of each.
    /// The title's copy, one record more included, stays as written.
    /// `None` without a host GPU.
    fn clamped_words(
        kind: u32,
        words: &[u32],
        records: u32,
        calls: u32,
        fields: usize,
    ) -> Option<Vec<u32>> {
        let host = super::host()?;
        // Words in a record of `kind`.
        let per = match kind {
            0 => 4,
            1 => 5,
            _ => 3,
        };
        let spec = (kind, per as u32 * 4, records);
        let all = per * (records as usize + 1);
        let (buffer, data) = host.shared_buffer(all as u64 * 4).unwrap();
        let data = data.cast::<u32>();
        let list = host.new_list().unwrap();
        // SAFETY: our mapped buffer, written before the list runs and read
        // after it has; a live buffer.
        unsafe {
            for at in 0..all {
                data.add(at).write_volatile(words[at % words.len()]);
            }
            let gpu = buffer.GetGPUVirtualAddress();
            let mut out = None;
            for _ in 0..calls {
                let clamped = super::bound(&list, gpu, spec).unwrap();
                super::done(&list, &clamped);
                out = Some(clamped);
            }
            list.close();
            host.execute(&[&list]);
            for at in 0..all {
                assert_eq!(data.add(at).read_volatile(), words[at % words.len()], "the title's copy as written");
            }
            let mut out_data = std::ptr::null_mut();
            out?.Map(0, None, Some(&mut out_data)).unwrap();
            assert!(fields <= per);
            let indices = (0..records as usize)
                .flat_map(|record| (0..fields).map(move |field| record * per + field));
            Some(indices.map(|i| out_data.cast::<u32>().add(i).read_volatile()).collect())
        }
    }

    #[test]
    fn mesh_clamp_linearizes_in_budget_grids_and_charges_all_axes() {
        use super::D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;
        let mesh = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH.0 as u32;
        let words = [3, 2, 4, 3, 0, 4, 3, 2, 0, 65_535, 1, 1, u32::MAX, 2, 3, 257, 256, 1];
        let Some(grids) = clamped_words(mesh, &words, 6, 1, 3) else {
            return; // no host GPU here
        };
        // Each of these grids costs 65,532 mesh groups. The list has
        // room for 16, so the last two records must run empty.
        let budget = clamped(mesh, &[127, 172, 3], 18, 1).unwrap();
        assert_eq!(
            (grids, budget),
            (
                vec![24, 1, 1, 0, 1, 1, 0, 1, 1, 65_535, 1, 1, 65_535, 1, 1, 257, 1, 1],
                [65_532; 16].into_iter().chain([0; 2]).collect(),
            ),
            "linear group counts, oversized fallback and list budget",
        );
    }

    /// Rejected records do not wrap the budget: of 128 garbage draws (2^26
    /// vertex instances each once clamped) two run, where a wrapping add
    /// let four; of 600 garbage mesh dispatches 16, not 31.
    #[test]
    fn rejected_records_do_not_wrap_the_budget() {
        let Some(draws) = clamped(0, &[u32::MAX], 128, 1) else {
            return; // no host GPU here
        };
        assert_eq!(draws.iter().filter(|&&count| count != 0).count(), 2);
        let meshes = clamped(10, &[u32::MAX], 600, 1).unwrap();
        assert_eq!(meshes.iter().filter(|&&groups| groups != 0).count(), 16);
    }

    /// The budget goes to the first records, in order: of 200 draws of
    /// 3 * 2^18 vertices the first 170 run (it ends inside the third tile);
    /// small records that all fit all run.
    #[test]
    fn the_budget_goes_to_the_first_records() {
        let Some(draws) = clamped(0, &[3 << 18, 1, 0, 0], 200, 1) else {
            return; // no host GPU here
        };
        assert!(draws[..170].iter().all(|&count| count != 0));
        assert!(draws[170..].iter().all(|&count| count == 0));
        let small = clamped(0, &[1], 1 << 16, 1).unwrap();
        assert!(small.iter().all(|&count| count == 1));
    }

    /// A record that does not fit ends the budget: after two draws of
    /// 3 * 2^24 vertex instances a third takes what is left, and the small
    /// ones after it run empty.
    #[test]
    fn a_record_that_does_not_fit_ends_the_budget() {
        let mut words = [1 << 24, 3, 0, 0].repeat(3);
        words.extend([3, 1, 0, 0].repeat(6));
        let Some(draws) = clamped(0, &words, 8, 1) else {
            return; // no host GPU here
        };
        let ran: Vec<bool> = draws.iter().map(|&count| count != 0).collect();
        assert_eq!(ran, [true, true, false, false, false, false, false, false]);
    }

    /// Two lists clamping into their own argument buffers, run in either
    /// order: every barrier starts from the buffer's current state (the
    /// debug layer's rule; runs light19 and light21 had the title's GPU
    /// heap moved to unordered access under the input assembler), the
    /// title's copy is never transitioned, and all end common.
    #[test]
    fn argument_buffers_move_from_their_current_states() {
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (title, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: our mapped buffer; a live buffer.
        let gpu = unsafe {
            data.cast::<[u32; 3]>().write_volatile([70_000, 1, 1]);
            title.GetGPUVirtualAddress()
        };
        let key = |r: &super::ID3D12Resource| r.as_raw() as usize;
        for a_first in [true, false] {
            let lists = [host.new_list().unwrap(), host.new_list().unwrap()];
            issued();
            let recorded: Vec<_> = lists
                .iter()
                .map(|l| {
                    let out = super::bound(l, gpu, (2, 12, 1)).unwrap();
                    super::done(l, &out);
                    l.close();
                    (out, issued())
                })
                .collect();
            let order = if a_first { [0, 1] } else { [1, 0] };
            host.execute(&[&lists[order[0]], &lists[order[1]]]);
            let mut run = issued();
            run.extend(order.iter().flat_map(|&i| recorded[i].1.clone()));
            let mut states: std::collections::HashMap<usize, u32> =
                [key(&title), key(&recorded[0].0), key(&recorded[1].0)].into_iter().map(|k| (k, 0)).collect();
            replay(&run, &mut states).unwrap();
            assert!(run.iter().all(|t| t.0 != key(&title)), "the title's copy is never transitioned");
            assert!(states.values().all(|&s| s == 0), "all common again");
            for (out, _) in &recorded {
                let mut p = std::ptr::null_mut();
                // SAFETY: a mapped buffer of ours, the lists have run.
                let clamped = unsafe {
                    out.Map(0, None, Some(&mut p)).unwrap();
                    p.cast::<[u32; 3]>().read_volatile()
                };
                assert_eq!(clamped, [65_535, 1, 1]);
            }
            // SAFETY: as above.
            assert_eq!(unsafe { data.cast::<[u32; 3]>().read_volatile() }, [70_000, 1, 1], "the title's copy as written");
        }
    }

    /// The budget is the list's: a second execution of the same records
    /// finds it spent, and they all run empty.
    #[test]
    fn a_spent_budget_stays_spent() {
        let Some(twice) = clamped(1, &[u32::MAX], 64, 2) else {
            return; // no host GPU here
        };
        assert!(twice.iter().all(|&count| count == 0));
    }

    #[test]
    fn batch_argument_clamp_builds() {
        if super::host().is_none() {
            return;
        }
        assert!(super::batch_clamp().is_some());
    }

    /// Reads `words` 32-bit words from `buf` at byte offset `at` after execution.
    fn read_words(buf: &super::ID3D12Resource, at: u64, words: usize) -> Vec<u32> {
        let mut p = std::ptr::null_mut();
        // SAFETY: mapped host-visible shared scratch buffer after GPU completion.
        unsafe {
            buf.Map(0, None, Some(&mut p)).unwrap();
            let base = p.cast::<u8>().add(at as usize).cast::<u32>();
            (0..words).map(|i| base.add(i).read_volatile()).collect()
        }
    }

    /// Hoisted batched clamping matches inline per-draw clamping word-for-word
    /// across >64 single-record draws, mixed kinds (`0, 1, 2, 10`), a multi-
    /// record (>64) draw, and mid-wave `BUDGET` exhaustion — while leaving the
    /// title's source buffer completely unmodified.
    #[test]
    fn hoisted_clamps_match_inline_bounds_and_budget_order() {
        let Some(host) = super::host() else {
            return;
        };
        const SRC_BYTES: usize = 64 << 10;
        let (src, src_ptr) = host.shared_buffer(SRC_BYTES as u64).unwrap();
        let mut initial = vec![0xA5u8; SRC_BYTES];
        let put = |buf: &mut [u8], off: usize, v: u32| {
            buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
        };

        struct Spec {
            off: u64,
            kind: u32,
            stride: u32,
            records: u32,
        }
        let mut specs = Vec::new();
        let mut cur = 64usize;
        // 75 consecutive single-record jobs (spanning >1 full 64-lane wave).
        for i in 0..75u32 {
            let (kind, stride) = match i % 4 {
                0 => (0u32, 16u32),
                1 => (1u32, 20u32),
                2 => (2u32, 12u32),
                _ => (10u32, 12u32),
            };
            match kind {
                0 => {
                    put(&mut initial, cur, 5000 + i * 11);
                    put(&mut initial, cur + 4, 3);
                    put(&mut initial, cur + 8, 10 + i);
                    put(&mut initial, cur + 12, 20 + i);
                }
                1 => {
                    put(&mut initial, cur, 7000 + i * 13);
                    put(&mut initial, cur + 4, 2);
                    put(&mut initial, cur + 8, 100 + i);
                    put(&mut initial, cur + 12, (-30i32 + i as i32) as u32);
                    put(&mut initial, cur + 16, 200 + i);
                }
                2 => {
                    put(&mut initial, cur, 10 + (i % 7));
                    put(&mut initial, cur + 4, 2);
                    put(&mut initial, cur + 8, 1);
                }
                _ => {
                    // First mesh tests 65,535 clamp; remaining stay small so BUDGET crosses in the tail.
                    put(&mut initial, cur, if i == 3 { 70_000 } else { 8 + (i % 5) });
                    put(&mut initial, cur + 4, 4);
                    put(&mut initial, cur + 8, 3);
                }
            }
            specs.push(Spec {
                off: cur as u64,
                kind,
                stride,
                records: 1,
            });
            cur += stride as usize;
        }
        // Multi-record (>64) indexed draw.
        let multi_off = cur;
        for r in 0..70u32 {
            let b = multi_off + (r * 20) as usize;
            put(&mut initial, b, 3000 + r);
            put(&mut initial, b + 4, 2);
            put(&mut initial, b + 8, r);
            put(&mut initial, b + 12, r * 2);
            put(&mut initial, b + 16, r * 3);
        }
        specs.push(Spec {
            off: multi_off as u64,
            kind: 1,
            stride: 20,
            records: 70,
        });
        cur += 70 * 20;
        // 80 heavy indexed draws that exhaust BUDGET mid-wave (4,000,000 each).
        for i in 0..80u32 {
            put(&mut initial, cur, 1_000_000);
            put(&mut initial, cur + 4, 4);
            put(&mut initial, cur + 8, 700 + i);
            put(&mut initial, cur + 12, 800 + i);
            put(&mut initial, cur + 16, 900 + i);
            specs.push(Spec {
                off: cur as u64,
                kind: 1,
                stride: 20,
                records: 1,
            });
            cur += 20;
        }

        // SAFETY: mapped `SRC_BYTES` shared buffer; live buffer.
        let base_gpu = unsafe {
            std::ptr::copy_nonoverlapping(initial.as_ptr(), src_ptr, SRC_BYTES);
            src.GetGPUVirtualAddress()
        };

        // Hoisted path.
        let list_hoisted = host.new_list().unwrap();
        let mut hoisted_locs = Vec::with_capacity(specs.len());
        for s in &specs {
            let loc = super::hoist(
                &list_hoisted,
                &src,
                base_gpu + s.off,
                (s.kind, s.stride, s.records),
            )
            .unwrap();
            hoisted_locs.push(loc);
        }
        list_hoisted.close();
        host.execute(&[&list_hoisted]);

        // Inline reference path (one submission per readback, sharing one list budget).
        let list_inline = host.new_list().unwrap();
        let (rb_inline, rb_ptr) = host.shared_buffer(SRC_BYTES as u64).unwrap();
        let mut rb_offsets = Vec::with_capacity(specs.len());
        let mut rb_cur = 0u64;
        for s in &specs {
            let out = super::bound(&list_inline, base_gpu + s.off, (s.kind, s.stride, s.records)).unwrap();
            let bytes = u64::from(s.stride * s.records);
            // SAFETY: open list; `out` is in `INDIRECT_ARGUMENT` after `bound`.
            unsafe {
                list_inline.list.ResourceBarrier(&[super::transition(
                    &out,
                    super::D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    super::D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                list_inline.list.CopyBufferRegion(&rb_inline, rb_cur, &out, 0, bytes);
                list_inline.list.ResourceBarrier(&[super::transition(
                    &out,
                    super::D3D12_RESOURCE_STATE_COPY_SOURCE,
                    super::D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                )]);
            }
            super::done(&list_inline, &out);
            rb_offsets.push((rb_cur, bytes as usize));
            rb_cur += bytes;
        }
        list_inline.close();
        host.execute(&[&list_inline]);

        // SAFETY: GPU completed; verify title copy untouched and outputs identical.
        let after_src = unsafe { std::slice::from_raw_parts(src_ptr.cast_const(), SRC_BYTES).to_vec() };
        assert_eq!(after_src, initial, "title argument buffer must never be modified");

        let mut tail_kept = 0usize;
        let mut tail_zeroed = 0usize;
        for (idx, ((dst, at), (rb_off, bytes))) in hoisted_locs.iter().zip(rb_offsets.iter()).enumerate() {
            let got = read_words(dst, *at, *bytes / 4);
            // SAFETY: within mapped `rb_inline`.
            let expected = unsafe {
                let p = rb_ptr.add(*rb_off as usize).cast::<u32>();
                (0..*bytes / 4).map(|i| p.add(i).read_volatile()).collect::<Vec<_>>()
            };
            assert_eq!(got, expected, "job {idx} hoisted output must match inline bound()");
            if idx == 7 {
                // All 10 x 4 x 3 groups get distinct translated indices.
                assert_eq!(got, [120, 1, 1]);
            }
            if idx >= 76 {
                if got[0] == 1_000_000 {
                    tail_kept += 1;
                } else {
                    assert_eq!(got[0], 0);
                    tail_zeroed += 1;
                }
            }
        }
        assert!(tail_kept > 0 && tail_zeroed > 0, "tail must cross BUDGET mid-wave");
    }

    /// Multiple source buffers and >256 KiB chunk rollover preserve strict
    /// draw-order budget saturation across dispatch runs and reuse both chunks
    /// and upload buffer across `HostList::reset()`.
    #[test]
    fn hoisted_clamps_across_multiple_buffers_and_chunks_preserve_budget_order() {
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return;
        };
        let list = host.new_list().unwrap();
        let (src_a, ptr_a) = host.shared_buffer(4096).unwrap();
        let (src_b, ptr_b) = host.shared_buffer(4096).unwrap();
        // SAFETY: mapped 4096-byte shared buffers; live buffers.
        let (gpu_a, gpu_b) = unsafe {
            let write = |p: *mut u8, off: usize, count: u32, inst: u32, tag: u32| {
                let w = p.add(off).cast::<u32>();
                w.write(count);
                w.add(1).write(inst);
                w.add(2).write(tag);
                w.add(3).write(0);
                w.add(4).write(0);
            };
            write(ptr_a, 0, 0, 1, 0xAAAA);
            write(ptr_a, 32, 1 << 22, 4, 0x1111); // BUDGET / 8
            write(ptr_b, 32, 1 << 22, 4, 0x2222); // BUDGET / 8
            (src_a.GetGPUVirtualAddress(), src_b.GetGPUVirtualAddress())
        };

        for cycle in 0..2u32 {
            let mut probes = Vec::new();
            for i in 0..4u32 {
                let (s, g) = if i % 2 == 0 { (&src_a, gpu_a) } else { (&src_b, gpu_b) };
                probes.push(super::hoist(&list, s, g + 32, (1, 20, 1)).unwrap());
            }
            // Spill across 256 KiB chunk boundary (8,192 * 32B = 256 KiB).
            let mut spilled = false;
            let first_chunk = probes[0].0.as_raw();
            for _ in 0..8_250 {
                let (c, _) = super::hoist(&list, &src_a, gpu_a, (1, 20, 1)).unwrap();
                if c.as_raw() != first_chunk {
                    spilled = true;
                }
            }
            assert!(spilled, "must span at least two 256 KiB hoisted scratch chunks");
            for i in 0..6u32 {
                let (s, g) = if i % 2 == 0 { (&src_a, gpu_a) } else { (&src_b, gpu_b) };
                probes.push(super::hoist(&list, s, g + 32, (1, 20, 1)).unwrap());
            }
            list.close();
            host.execute(&[&list]);

            for (idx, (dst, at)) in probes.iter().enumerate() {
                let w = read_words(dst, *at, 5);
                if idx < 8 {
                    assert_eq!((w[0], w[1]), (1 << 22, 4), "cycle {cycle} probe {idx} must fit");
                } else {
                    assert_eq!(w[0], 0, "cycle {cycle} probe {idx} must zero past BUDGET");
                }
            }
            list.reset();
        }
    }

    /// Replays all transition barriers emitted by two hoisted lists run in
    /// either order (matching `argument_buffers_move_from_their_current_states`):
    /// the title's buffer is never transitioned and every hoisted scratch chunk
    /// moves `COMMON -> UNORDERED_ACCESS -> INDIRECT_ARGUMENT -> COMMON`.
    #[test]
    fn hoisted_transitions_start_from_current_states_and_never_touch_title_buffer() {
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return;
        };
        let (title, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: mapped buffer; live buffer.
        let gpu = unsafe {
            data.cast::<[u32; 3]>().write_volatile([70_000, 1, 1]);
            title.GetGPUVirtualAddress()
        };
        let key = |r: &super::ID3D12Resource| r.as_raw() as usize;
        for a_first in [true, false] {
            let lists = [host.new_list().unwrap(), host.new_list().unwrap()];
            issued();
            let recorded: Vec<_> = lists
                .iter()
                .map(|l| {
                    let (out, at) = super::hoist(l, &title, gpu, (2, 12, 1)).unwrap();
                    l.close();
                    (out, at, issued())
                })
                .collect();
            let order = if a_first { [0, 1] } else { [1, 0] };
            let mut states: std::collections::HashMap<usize, u32> =
                [key(&title), key(&recorded[0].0), key(&recorded[1].0)]
                    .into_iter()
                    .map(|k| (k, 0))
                    .collect();
            // Run each list one at a time so prelude + list barriers replay in exact GPU order.
            for &i in &order {
                issued();
                host.execute(&[&lists[i]]);
                let mut run = issued();
                run.extend_from_slice(&recorded[i].2);
                replay(&run, &mut states).unwrap();
                assert!(run.iter().all(|t| t.0 != key(&title)), "title copy must never be transitioned");
            }
            assert!(states.values().all(|&s| s == 0), "all buffers must return to COMMON");
            for (out, at, _) in &recorded {
                assert_eq!(read_words(out, *at, 3), vec![65_535, 1, 1]);
            }
            // SAFETY: mapped title buffer untouched.
            assert_eq!(unsafe { data.cast::<[u32; 3]>().read_volatile() }, [70_000, 1, 1]);
        }
    }

    /// Re-submitting the same closed hoisted list twice re-clamps cleanly into
    /// its own scratch chunks without mutating the title's source buffer.
    #[test]
    fn hoisted_list_resubmitted_twice_reclamps_cleanly() {
        let Some(host) = super::host() else {
            return;
        };
        let list = host.new_list().unwrap();
        let (src, ptr) = host.shared_buffer(256).unwrap();
        // SAFETY: mapped buffer; live buffer.
        let gpu = unsafe {
            let p = ptr.cast::<u32>();
            p.write(u32::MAX);
            p.add(1).write(u32::MAX);
            p.add(2).write(42);
            p.add(3).write(7);
            p.add(4).write(99);
            src.GetGPUVirtualAddress()
        };
        let (dst, at) = super::hoist(&list, &src, gpu, (1, 20, 1)).unwrap();
        list.close();
        for _ in 0..2 {
            host.execute(&[&list]);
            assert_eq!(read_words(&dst, at, 5), vec![1 << 24, 4, 42, 7, 99]);
            // SAFETY: title buffer untouched.
            assert_eq!(unsafe { ptr.cast::<u32>().read_volatile() }, u32::MAX);
        }
    }

    /// An earlier list's compute write to a title argument buffer is visible
    /// to the hoisted prelude of the following list.
    #[test]
    fn hoisted_prelude_orders_after_earlier_list_compute_write() {
        let Some(host) = super::host() else {
            return;
        };
        const WRITER: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer dst : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    dst.Store3(0, uint3(90000, 1, 1));
}
"#;
        let writer = super::compile(WRITER, super::s!("writer")).unwrap();
        let (src, ptr) = host.shared_buffer(256).unwrap();
        // SAFETY: mapped buffer; live buffer.
        let gpu = unsafe {
            std::ptr::write_bytes(ptr, 0, 256);
            src.GetGPUVirtualAddress()
        };
        let write_list = host.new_list().unwrap();
        // SAFETY: open list, live compute pipeline and UAV buffer.
        unsafe {
            write_list.list.SetComputeRootSignature(&writer.root);
            write_list.list.SetPipelineState(&writer.pso);
            write_list.list.SetComputeRootUnorderedAccessView(0, gpu);
            write_list.list.Dispatch(1, 1, 1);
        }
        write_list.close();

        let hoisted_list = host.new_list().unwrap();
        let (dst, at) = super::hoist(&hoisted_list, &src, gpu, (2, 12, 1)).unwrap();
        hoisted_list.close();

        host.execute(&[&write_list, &hoisted_list]);
        assert_eq!(read_words(&dst, at, 3), vec![65_535, 1, 1]);
    }

    /// Release benchmark comparing inline `bound()` + `done()` vs prelude-
    /// hoisted batched clamping across 9,000 indirect indexed draws.
    ///
    /// Run with:
    /// `cargo test --release -p victory-gpu --locked -- --ignored --nocapture benchmark_hoisted_vs_inline_indirect_clamps`
    #[test]
    #[ignore = "benchmark"]
    fn benchmark_hoisted_vs_inline_indirect_clamps() {
        let Some(host) = super::host() else {
            return;
        };
        let list_inline = host.new_list().unwrap();
        let list_hoisted = host.new_list().unwrap();
        const DRAWS: usize = 9_000;
        const STRIDE: u32 = 20;
        let (src, ptr) = host.shared_buffer((DRAWS * STRIDE as usize) as u64).unwrap();
        // SAFETY: mapped buffer; live buffer.
        let base_gpu = unsafe {
            for i in 0..DRAWS {
                let p = ptr.add(i * STRIDE as usize).cast::<u32>();
                p.write(300);
                p.add(1).write(1);
                p.add(2).write(0);
                p.add(3).write(0);
                p.add(4).write(0);
            }
            src.GetGPUVirtualAddress()
        };

        let run_inline = || {
            for i in 0..DRAWS {
                let out = super::bound(&list_inline, base_gpu + (i * STRIDE as usize) as u64, (1, STRIDE, 1)).unwrap();
                super::done(&list_inline, &out);
            }
            list_inline.close();
            host.execute(&[&list_inline]);
            list_inline.reset();
        };
        let run_hoisted = || {
            for i in 0..DRAWS {
                let _ = super::hoist(&list_hoisted, &src, base_gpu + (i * STRIDE as usize) as u64, (1, STRIDE, 1)).unwrap();
            }
            list_hoisted.close();
            host.execute(&[&list_hoisted]);
            list_hoisted.reset();
        };

        for _ in 0..3 {
            run_inline();
            run_hoisted();
        }
        const ITERS: u32 = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            run_inline();
        }
        let inline_ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(ITERS);

        let t1 = std::time::Instant::now();
        for _ in 0..ITERS {
            run_hoisted();
        }
        let hoisted_ms = t1.elapsed().as_secs_f64() * 1e3 / f64::from(ITERS);

        eprintln!(
            "{DRAWS} indirect draw clamps: inline {inline_ms:.2} ms/frame vs hoisted {hoisted_ms:.2} ms/frame ({:.1}x)",
            inline_ms / hoisted_ms.max(1e-6)
        );
    }
}
