//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTL4LinkedFunctions.hpp
//
// Copyright 2020-2025 Apple Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#pragma once

#include "../Foundation/Foundation.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"

namespace MTL4
{
class LinkedFunctions;

class LinkedFunctions : public NS::Copying<LinkedFunctions>
{
public:
    static LinkedFunctions* alloc();

    NS::Array*              binaryFunctions() const;

    NS::Array*              functionDescriptors() const;

    NS::Dictionary*         groups() const;

    LinkedFunctions*        init();

    NS::Array*              privateFunctionDescriptors() const;

    void                    setBinaryFunctions(const NS::Array* binaryFunctions);

    void                    setFunctionDescriptors(const NS::Array* functionDescriptors);

    void                    setGroups(const NS::Dictionary* groups);

    void                    setPrivateFunctionDescriptors(const NS::Array* privateFunctionDescriptors);
};

}
_MTL_INLINE MTL4::LinkedFunctions* MTL4::LinkedFunctions::alloc()
{
    return NS::Object::alloc<MTL4::LinkedFunctions>(_MTL_PRIVATE_CLS(MTL4LinkedFunctions));
}

_MTL_INLINE NS::Array* MTL4::LinkedFunctions::binaryFunctions() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(binaryFunctions));
}

_MTL_INLINE NS::Array* MTL4::LinkedFunctions::functionDescriptors() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(functionDescriptors));
}

_MTL_INLINE NS::Dictionary* MTL4::LinkedFunctions::groups() const
{
    return Object::sendMessage<NS::Dictionary*>(this, _MTL_PRIVATE_SEL(groups));
}

_MTL_INLINE MTL4::LinkedFunctions* MTL4::LinkedFunctions::init()
{
    return NS::Object::init<MTL4::LinkedFunctions>();
}

_MTL_INLINE NS::Array* MTL4::LinkedFunctions::privateFunctionDescriptors() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(privateFunctionDescriptors));
}

_MTL_INLINE void MTL4::LinkedFunctions::setBinaryFunctions(const NS::Array* binaryFunctions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBinaryFunctions_), binaryFunctions);
}

_MTL_INLINE void MTL4::LinkedFunctions::setFunctionDescriptors(const NS::Array* functionDescriptors)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFunctionDescriptors_), functionDescriptors);
}

_MTL_INLINE void MTL4::LinkedFunctions::setGroups(const NS::Dictionary* groups)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setGroups_), groups);
}

_MTL_INLINE void MTL4::LinkedFunctions::setPrivateFunctionDescriptors(const NS::Array* privateFunctionDescriptors)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPrivateFunctionDescriptors_), privateFunctionDescriptors);
}
