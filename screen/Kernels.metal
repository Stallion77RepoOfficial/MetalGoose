#if 0  // Disabled: replaced by Shaders.metal kernels to avoid duplicate symbols
#include <metal_stdlib>
using namespace metal;

kernel void integer_nearest_upscale(
    texture2d<float,  access::read>  src [[texture(0)]],
    texture2d<float,  access::write> dst [[texture(1)]],
    constant uint2& scale [[buffer(0)]],
    constant uint2& offset [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
)
{
    uint2 dst_coord = gid;
    if (dst_coord.x >= dst.get_width() || dst_coord.y >= dst.get_height()) {
        return;
    }

    // Check if pixel lies outside the scaled region
    // scaled region in dst: from offset to offset + scale * src dims
    uint2 scaled_region_min = offset;
    uint2 scaled_region_max = offset + scale * uint2(src.get_width(), src.get_height());

    if (dst_coord.x < scaled_region_min.x || dst_coord.y < scaled_region_min.y ||
        dst_coord.x >= scaled_region_max.x || dst_coord.y >= scaled_region_max.y)
    {
        // Write opaque black (0,0,0,255)
        dst.write(float4(0.0, 0.0, 0.0, 1.0), dst_coord);
        return;
    }

    // Compute source coordinate
    uint2 src_coord = (dst_coord - offset) / scale;

    src_coord.x = clamp(src_coord.x, 0u, src.get_width() - 1);
    src_coord.y = clamp(src_coord.y, 0u, src.get_height() - 1);

    float4 pixel = src.read(src_coord);
    dst.write(pixel, dst_coord);
}


inline float4 sample_bicubic_catmullrom(
    texture2d<float, access::sample> tex,
    sampler smp,
    float2 uv,
    uint2 texSize
)
{
    // Catmull-Rom kernel weights for 4 samples along one axis
    // Given fractional part f, weights:
    // w0 = -0.5 * f^3 + f^2 - 0.5 * f
    // w1 = 1.5 * f^3 - 2.5 * f^2 + 1
    // w2 = -1.5 * f^3 + 2 * f^2 + 0.5 * f
    // w3 = 0.5 * f^3 - 0.5 * f^2

    float2 texSizeF = float2(texSize);
    float2 pos = uv * texSizeF - 0.5;

    int2 iPos = int2(floor(pos));
    float2 f = pos - floor(pos);

    // Clamp indices for edge handling
    // We'll sample iPos.x-1,iPos.x,iPos.x+1,iPos.x+2 and same for y

    int x0 = clamp(iPos.x - 1, 0, int(texSize.x) - 1);
    int x1 = clamp(iPos.x    , 0, int(texSize.x) - 1);
    int x2 = clamp(iPos.x + 1, 0, int(texSize.x) - 1);
    int x3 = clamp(iPos.x + 2, 0, int(texSize.x) - 1);

    int y0 = clamp(iPos.y - 1, 0, int(texSize.y) - 1);
    int y1 = clamp(iPos.y    , 0, int(texSize.y) - 1);
    int y2 = clamp(iPos.y + 1, 0, int(texSize.y) - 1);
    int y3 = clamp(iPos.y + 2, 0, int(texSize.y) - 1);

    float fx = f.x;
    float fy = f.y;

    // Catmull-Rom weights for x
    float wx0 = -0.5*fx*fx*fx + fx*fx - 0.5*fx;
    float wx1 = 1.5*fx*fx*fx - 2.5*fx*fx + 1.0;
    float wx2 = -1.5*fx*fx*fx + 2.0*fx*fx + 0.5*fx;
    float wx3 = 0.5*fx*fx*fx - 0.5*fx*fx;

    // Catmull-Rom weights for y
    float wy0 = -0.5*fy*fy*fy + fy*fy - 0.5*fy;
    float wy1 = 1.5*fy*fy*fy - 2.5*fy*fy + 1.0;
    float wy2 = -1.5*fy*fy*fy + 2.0*fy*fy + 0.5*fy;
    float wy3 = 0.5*fy*fy*fy - 0.5*fy*fy;

    // Sample 16 texels
    float4 c00 = tex.read(uint2(x0, y0));
    float4 c10 = tex.read(uint2(x1, y0));
    float4 c20 = tex.read(uint2(x2, y0));
    float4 c30 = tex.read(uint2(x3, y0));

    float4 c01 = tex.read(uint2(x0, y1));
    float4 c11 = tex.read(uint2(x1, y1));
    float4 c21 = tex.read(uint2(x2, y1));
    float4 c31 = tex.read(uint2(x3, y1));

    float4 c02 = tex.read(uint2(x0, y2));
    float4 c12 = tex.read(uint2(x1, y2));
    float4 c22 = tex.read(uint2(x2, y2));
    float4 c32 = tex.read(uint2(x3, y2));

    float4 c03 = tex.read(uint2(x0, y3));
    float4 c13 = tex.read(uint2(x1, y3));
    float4 c23 = tex.read(uint2(x2, y3));
    float4 c33 = tex.read(uint2(x3, y3));

    // Interpolate along x
    float4 col0 = c00 * wx0 + c10 * wx1 + c20 * wx2 + c30 * wx3;
    float4 col1 = c01 * wx0 + c11 * wx1 + c21 * wx2 + c31 * wx3;
    float4 col2 = c02 * wx0 + c12 * wx1 + c22 * wx2 + c32 * wx3;
    float4 col3 = c03 * wx0 + c13 * wx1 + c23 * wx2 + c33 * wx3;

    // Interpolate along y
    float4 color = col0 * wy0 + col1 * wy1 + col2 * wy2 + col3 * wy3;

    return color;
}

kernel void easu_upscale(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant float2 &scale [[buffer(0)]],
    constant float2 &offset [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
)
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) {
        return;
    }

    float2 dst_size = float2(dst.get_width(), dst.get_height());
    float2 src_size = float2(src.get_width(), src.get_height());

    // Compute source UV normalized to [0,1]
    float2 src_uv = (float2(gid) - offset) / scale / src_size;

    // Outside source area => output black
    if (src_uv.x < 0.0 || src_uv.x > 1.0 || src_uv.y < 0.0 || src_uv.y > 1.0) {
        dst.write(float4(0.0, 0.0, 0.0, 1.0), gid);
        return;
    }

    // Use a sampler with clamp_to_edge and nearest filtering (we do manual bicubic so use nearest)
    sampler smp(address::clamp_to_edge, filter::nearest);

    float4 color = sample_bicubic_catmullrom(src, smp, src_uv, uint2(src.get_width(), src.get_height()));

    // Clamp color to [0,1]
    color = clamp(color, 0.0, 1.0);

    dst.write(color, gid);
}

kernel void rcas_sharpen(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant float &sharpness [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
)
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) {
        return;
    }

    uint width = src.get_width();
    uint height = src.get_height();

    int2 center = int2(gid);
    int2 up    = int2(gid.x, max(int(gid.y) - 1, 0));
    int2 down  = int2(gid.x, min(int(gid.y) + 1, int(height) - 1));
    int2 left  = int2(max(int(gid.x) - 1, 0), int(gid.y));
    int2 right = int2(min(int(gid.x) + 1, int(width) - 1), int(gid.y));

    float4 c = src.read(uint2(center));
    float4 u = src.read(uint2(up));
    float4 d = src.read(uint2(down));
    float4 l = src.read(uint2(left));
    float4 r = src.read(uint2(right));

    float4 neighbors_avg = (u + d + l + r) * 0.25;
    float4 detail = c - neighbors_avg;
    float4 out_color = c + sharpness * detail;

    out_color = clamp(out_color, 0.0, 1.0);

    dst.write(out_color, gid);
}

#endif // Disabled file
