#!/usr/bin/env bash

# 1. What shared libraries actually ship with the SDK?
ls -la /home/rudycazabon/Libraries/vulkan-sdk/1.4.350.1/x86_64/lib/

# 2. Which .so actually exports gfxCreateDevice?
for f in /home/rudycazabon/Libraries/vulkan-sdk/1.4.350.1/x86_64/lib/*.so*; do
  echo "== $f =="
  nm -D "$f" 2>/dev/null | grep -i gfxCreateDevice
done

# 3. Confirm which library CMake actually picked up for SLANG_LIBRARY
grep -i "SLANG_LIBRARY" build/CMakeCache.txt