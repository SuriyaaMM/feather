#include "managers.hpp"
#include "foundation.hpp"
#include <ranges>

namespace feather
{
namespace vulkan
{

DeviceManager::DeviceManager(vk::Instance &rInstance) :
		m_instanceRef(rInstance), m_physicalDevice(std::nullptr_t()),
		m_logicalDevice(std::nullptr_t())
{
	std::vector<vk::PhysicalDevice> physicalDevices = m_instanceRef.enumeratePhysicalDevices();

	for (const auto &[i, physicalDevice] : std::views::enumerate(physicalDevices)) {
		std::vector<vk::ExtensionProperties> availableExtensions =
			physicalDevice.enumerateDeviceExtensionProperties();
		std::vector<vk::LayerProperties> availableLayers =
			physicalDevice.enumerateDeviceLayerProperties();

		// Dump them to file
		FEATHER_ACCELERATOR_VULKAN_DUMP_EXTENSIONS_AND_LAYERS_TO(
			FEATHER_ACCELERATOR_VULKAN_DEVICE_EXTENSION_ENUMERATION_FILE(i),
			FEATHER_ACCELERATOR_VULKAN_DEVICE_LAYER_ENUMERATION_FILE(i), availableExtensions,
			availableLayers);
	}
}
} // namespace vulkan
} // namespace feather