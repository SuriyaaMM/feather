
unset VULKAN_SDK
unset VK_ADD_LAYER_PATH

# set vulkan variables
export VULKAN_SDK=$HOME/vulkan/1.4.328.1/x86_64

export PATH=$VULKAN_SDK/bin:$PATH
export LD_LIBRARY_PATH=$VULKAN_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PKG_CONFIG_PATH=$VULKAN_SDK/share/pkgconfig:$VULKAN_SDK/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}

# meson build commands
CXX=clang++ meson setup --wipe builddir
