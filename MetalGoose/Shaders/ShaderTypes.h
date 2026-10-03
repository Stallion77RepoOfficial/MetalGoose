// Types and constants shared by the Metal shaders and Swift (through the bridging header).

#ifndef MGShaderTypes_h
#define MGShaderTypes_h

typedef struct {
    float sharpness;
} SharpenParams;

typedef struct {
    float threshold;
    int maxSearchSteps;
} AntiAliasParams;

typedef struct {
    /// Where between the two captures the generated image sits, 0 at the first and 1 at the second.
    float phase;
    /// The change between the captures, as a share of the full range, at which the generated image is left alone.
    float tolerance;
} StabilityBlendParams;

/// The motion field is summarised over a fixed square grid of tiles, one thread per
/// tile, inside a single threadgroup. The Swift side dispatches exactly this many.
#define MG_MOTION_GRID 16

/// `blendTowardCaptures` evaluates a tile of pixels plus a halo in threadgroup memory. The
/// kernel's tile and the dispatch's threadgroup size are the same constants.
#define MG_BLEND_TILE_WIDTH 32
#define MG_BLEND_TILE_HEIGHT 8

#endif
