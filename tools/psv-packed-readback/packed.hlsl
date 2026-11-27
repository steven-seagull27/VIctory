// Original two-stage interpolator fixture. Unequal semantic indices share a register.
struct Varyings {
    float2 pair : TEXCOORD4;
    float tail : TEXCOORD1;
    float last : TEXCOORD7;
    noperspective float affine : TEXCOORD9;
    float4 position : SV_Position;
};

Varyings vs(uint vertex : SV_VertexID) {
    float u = (vertex << 1) & 2;
    float v = vertex & 2;
    float w = 1.0;
#ifdef VARIABLE_W
    w = vertex == 0 ? 1.0 : vertex == 1 ? 2.0 : 4.0;
#endif
    Varyings o;
    o.pair = float2(9.0 + u + v, 0.125 + 0.625*u + 0.03125*v);
    o.tail = 1.25 - 0.25*u + 0.125*v;
    o.last = 0.875 - 0.3125*u + 0.0625*v;
    o.affine = -0.5 + 0.375*u + 0.6875*v;
    o.position = float4((2*u - 1)*w, (1 - 2*v)*w, 0.5*w, w);
    return o;
}

float4 ps(Varyings v) : SV_Target {
    return float4(v.pair.y, v.tail, v.last, v.affine);
}
