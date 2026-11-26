[numthreads(THREADS, 1, 1)]
[WaveSize(WIDTH)]
void main(uint tid : SV_GroupIndex) {
    RWByteAddressBuffer output = ResourceDescriptorHeap[0];
    uint4 mask = WaveActiveBallot(tid >= 7);
    uint first = mask.x != 0 ? firstbitlow(mask.x)
                : mask.y != 0 ? 32 + firstbitlow(mask.y) : 0;
    output.Store4(32 * tid, uint4(tid, WaveGetLaneIndex(), WaveGetLaneCount(), mask.x));
    output.Store4(32 * tid + 16, uint4(mask.y, first, WaveReadLaneAt(100 + tid, first),
                                      WaveReadLaneAt(100 + tid, 7)));
}
