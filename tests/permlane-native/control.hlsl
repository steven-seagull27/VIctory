// Native D3D12 control for VIctory #1737. Compile with DXC, shader model 6.6.
cbuffer Inputs : register(b0) {
    uint4 selectorsExec; // selector low, selector high, EXEC low, EXEC high
};

[WaveSize(WIDTH)]
[numthreads(128, 1, 1)]
void main(uint thread : SV_GroupIndex) {
    RWByteAddressBuffer output = ResourceDescriptorHeap[0];
    uint lane = WaveGetLaneIndex();
    uint position = lane % 16;
    uint packed = position < 8 ? selectorsExec.x : selectorsExec.y;
    uint selected = (packed >> (4 * (position % 8))) & 15;
    uint source = ((lane & ~15u) ^ (CROSS * 16)) | selected;
    uint mask = lane < 32 ? selectorsExec.z : selectorsExec.w;
    bool enabled = ((mask >> (lane % 32)) & 1) != 0;
    uint4 ballot = WaveActiveBallot(enabled);
    uint sourceWord = source < 32 ? ballot.x : ballot.y;
    bool sourceEnabled = ((sourceWord >> (source % 32)) & 1) != 0;
    uint safe = (FI || sourceEnabled) ? source : lane;
    uint value = 0x12340000 + thread;
    uint old = ALIAS ? value : 42;
    uint sample = WaveReadLaneAt(value, safe);
    uint result = enabled ? ((FI || sourceEnabled) ? sample : (BC ? 0 : old)) : old;
#if DIAG
    // Include a direct read independently of the ballot and the index safeguard.
    output.Store4(48 * thread, uint4(lane, WaveGetLaneCount(), source, safe));
    output.Store4(48 * thread + 16,
                  uint4(ballot.x, ballot.y, sample, WaveReadLaneAt(value, source)));
    uint4 hardwareActive = WaveActiveBallot(true);
    output.Store4(48 * thread + 32, uint4(hardwareActive.xy, uint(sourceEnabled), uint(enabled)));
#else
    output.Store(4 * thread, result);
#endif
}
