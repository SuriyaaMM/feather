# Accelerators for Feather

## Vulkan

- Pre-Requisites
    - `C++23` (built using ```clang version 20.1.8 (Fedora 20.1.8-4.fc42), 
Target: x86_64-redhat-linux-gnu```)
    - `Meson` build system
    - `Vulkan SDK` (tested on 1.4.328.1)

```bash
# navigate to 
cd feather/accelerators

# for building
sh build.sh
# for compiling
sh compile.sh
# for running
sh run.sh
```

- Note: Vulkan Home Directory is set in `build.sh` under the variable `VULKAN_SDK`
