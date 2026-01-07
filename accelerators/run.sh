# unset existing ones
unset VULKAN_SDK
unset VK_LAYER_PATH

# set vulkan variables
export VULKAN_SDK=$HOME/vulkan/1.4.328.1/x86_64
export VK_ADD_LAYER_PATH=$VULKAN_SDK/share/vulkan/explicit_layer.d
export LD_LIBRARY_PATH=$VULKAN_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# makes debugging lot easier
# might get nvidia-driver leaks if not set detect_leaks=0
export ASAN_SYMBOLIZER_PATH=$(which llvm-symbolizer)
export ASAN_OPTIONS=detect_leaks=0:abort_on_error=1
export UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1

builddir/feather