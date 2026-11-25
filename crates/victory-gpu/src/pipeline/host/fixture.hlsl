float4 fullscreen(uint vertex : SV_VertexID) : SV_Position {
    float2 uv = float2((vertex << 1) & 2, vertex & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}

// Only the signature of this shader is used. VIctory emits the actual pixel program.
float4 color_signature() : SV_Target {
    return float4(1, 1, 1, 1);
}
