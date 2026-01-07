project('feather-acclerator-vulkan', 'cpp',
  version: '0.1.0',
  default_options: [
    'cpp_std=c++23',
    'b_sanitize=address,undefined',
    'b_lundef=false'
  ]
)

vulkan_dep = dependency('vulkan')

headers = files(
  'vk/managers.hpp',
  'vk/foundation.hpp'
)

add_project_arguments(
  '-DVULKAN_HPP_NO_STRUCT_CONSTRUCTORS', 
  language: 'cpp'
)

executable('feather',
  sources: [
    'vk/main.cpp',
    # managers
    'vk/InstanceManager.cpp',
    'vk/DeviceManager.cpp',
    headers
  ],
  dependencies: vulkan_dep
)
