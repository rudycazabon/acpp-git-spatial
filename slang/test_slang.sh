#!/usr/bin/env bash

SLANG_H=/home/rudycazabon/Libraries/vulkan-sdk/1.4.350.1/x86_64/include/slang/slang-gfx.h

# IShaderProgram::Desc — exact member names
grep -n "struct IShaderProgram" -A 30 "$SLANG_H" | head -50

# IDevice::createProgram signature
grep -n "createProgram" -A 5 "$SLANG_H"

# Command queue creation/fetching API
grep -n "class ICommandQueue" -A 40 "$SLANG_H" | head -60
grep -n "createCommandQueue\|getQueue\|getCommandQueue" "$SLANG_H"

# Fence / sync API
grep -n "class IFence\|ISyncFence\|createFence\|createSyncFence" -A 20 "$SLANG_H"

# Compute encoder bind/dispatch calls
grep -n "class IComputeCommandEncoder" -A 30 "$SLANG_H" | head -50

# Buffer view creation signature
grep -n "createBufferView" -A 6 "$SLANG_H"