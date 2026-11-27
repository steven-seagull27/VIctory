// The .y and .z components correspond to vertices 1 and 2 of this triangle list.
float4 dual(float3 p : SV_Barycentrics0, noperspective float3 a : SV_Barycentrics1) : SV_Target {
    return float4(p.yz, a.yz);
}
float4 perspective_only(float3 p : SV_Barycentrics0) : SV_Target {
    return float4(p.yz, p.yz);
}
float4 affine_only(noperspective float3 a : SV_Barycentrics0) : SV_Target {
    return float4(a.yz, a.yz);
}
