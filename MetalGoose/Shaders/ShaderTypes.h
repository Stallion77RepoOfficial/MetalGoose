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

/// The motion field is summarised over a fixed square grid of tiles, one thread per
/// tile, inside a single threadgroup. The Swift side dispatches exactly this many.
#define MG_MOTION_GRID 16

/// `staticMask` evaluates a tile of pixels plus a halo in threadgroup memory. The
/// kernel's tile and the dispatch's threadgroup size are the same constants.
#define MG_MASK_TILE_WIDTH 32
#define MG_MASK_TILE_HEIGHT 8

#endif
