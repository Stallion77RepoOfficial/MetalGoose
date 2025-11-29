#import <Metal/Metal.h>
#import "MetalCPPBridge.h"
#include "../metal-cpp/SingleHeader/Metal.hpp"

#include <string>

using namespace MTL;
using namespace MTLFX;
using namespace MTL4FX;

namespace
{
std::string gLastError;

[[noreturn]] void AbortWithReason(const char* message)
{
    gLastError.assign(message);
    NSLog(@"metal-cpp integration failure: %s", message);
    std::abort();
}

void Validate(bool condition, const char* message)
{
    if (!condition)
    {
        AbortWithReason(message);
    }
}
}

void MetalCPPEnsureReady(void)
{
    gLastError.clear();
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

    auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    Validate(device.get() != nullptr, "Failed to create Metal device via metal-cpp.");

    auto queue = NS::TransferPtr(device->newCommandQueue());
    Validate(queue.get() != nullptr, "Failed to create command queue via metal-cpp.");

    auto scalerDesc = NS::TransferPtr(MTLFX::SpatialScalerDescriptor::alloc()->init());
    Validate(scalerDesc.get() != nullptr, "Failed to allocate MetalFX spatial scaler descriptor.");
    scalerDesc->setInputWidth(16);
    scalerDesc->setInputHeight(16);
    scalerDesc->setOutputWidth(16);
    scalerDesc->setOutputHeight(16);
    scalerDesc->setColorTextureFormat(MTL::PixelFormatBGRA8Unorm);
    scalerDesc->setOutputTextureFormat(MTL::PixelFormatBGRA8Unorm);
    scalerDesc->setColorProcessingMode(MTLFX::SpatialScalerColorProcessingModePerceptual);

    const bool scalerSupportsMetal4 = scalerDesc->supportsMetal4FX(device.get());
    Validate(scalerSupportsMetal4, "Device does not advertise Metal 4 FX support for spatial scaling.");

    auto scaler = NS::TransferPtr(scalerDesc->newSpatialScaler(device.get()));
    Validate(scaler.get() != nullptr, "Failed to create MetalFX spatial scaler via metal-cpp.");

    auto interpolatorDesc = NS::TransferPtr(MTLFX::FrameInterpolatorDescriptor::alloc()->init());
    Validate(interpolatorDesc.get() != nullptr, "Failed to allocate MetalFX frame interpolator descriptor.");
    interpolatorDesc->setInputWidth(16);
    interpolatorDesc->setInputHeight(16);
    interpolatorDesc->setOutputWidth(16);
    interpolatorDesc->setOutputHeight(16);
    interpolatorDesc->setColorTextureFormat(MTL::PixelFormatBGRA8Unorm);
    interpolatorDesc->setOutputTextureFormat(MTL::PixelFormatBGRA8Unorm);
    interpolatorDesc->setDepthTextureFormat(MTL::PixelFormatDepth32Float);
    interpolatorDesc->setMotionTextureFormat(MTL::PixelFormatRG32Float);
    interpolatorDesc->setUITextureFormat(MTL::PixelFormatBGRA8Unorm);

    const bool interpolatorSupportsMetal4 = interpolatorDesc->supportsMetal4FX(device.get());
    Validate(interpolatorSupportsMetal4, "Device does not advertise Metal 4 FX support for frame interpolation.");

    auto interpolator = NS::TransferPtr(interpolatorDesc->newFrameInterpolator(device.get()));
    Validate(interpolator.get() != nullptr, "Failed to create MetalFX frame interpolator via metal-cpp.");

    auto temporalDesc = NS::TransferPtr(MTLFX::TemporalScalerDescriptor::alloc()->init());
    Validate(temporalDesc.get() != nullptr, "Failed to allocate MetalFX temporal scaler descriptor.");
    temporalDesc->setInputWidth(16);
    temporalDesc->setInputHeight(16);
    temporalDesc->setOutputWidth(16);
    temporalDesc->setOutputHeight(16);
    temporalDesc->setColorTextureFormat(MTL::PixelFormatBGRA8Unorm);
    temporalDesc->setOutputTextureFormat(MTL::PixelFormatBGRA8Unorm);
    temporalDesc->setMotionTextureFormat(MTL::PixelFormatRG32Float);
    temporalDesc->setDepthTextureFormat(MTL::PixelFormatDepth32Float);

    const bool temporalSupportsMetal4 = temporalDesc->supportsMetal4FX(device.get());
    Validate(temporalSupportsMetal4, "Device does not advertise Metal 4 FX support for temporal scaling.");

    auto temporalScaler = NS::TransferPtr(temporalDesc->newTemporalScaler(device.get()));
    Validate(temporalScaler.get() != nullptr, "Failed to create MetalFX temporal scaler via metal-cpp.");
}
