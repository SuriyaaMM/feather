#include "managers.hpp"
#include "foundation.hpp"

namespace feather
{
namespace vulkan
{
InstanceManager::InstanceManager() :
		m_instance(std::nullptr_t()), m_debugUtilsMessenger(std::nullptr_t())
{
	// API Version Information
	{
		const uint32_t apiVersion = vk::enumerateInstanceVersion();
		FEATHER_LOG("Vulkan API Version : " << apiVersion);
	}

	// Required Instance Extensions
	std::vector<const char *> requiredExtensions = {
		VK_EXT_DEBUG_UTILS_EXTENSION_NAME // for Debug Utils Messenger
	};
	// Required Instance Layers
	std::vector<const char *> requiredLayers = {
		"VK_LAYER_KHRONOS_validation" // for Debug Purposes
	};

	// Dump Extensions & Layer Properties to File
	{
		// Enumerate Extensions
		std::vector<vk::ExtensionProperties> availableExtensions =
			vk::enumerateInstanceExtensionProperties();
		// Enumerate Layers
		std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();

		FEATHER_ACCELERATOR_VULKAN_DUMP_EXTENSIONS_AND_LAYERS_TO(
			FEATHER_ACCELERATOR_VULKAN_INSTANCE_EXTENSION_ENUMERATION_FILE,
			FEATHER_ACCELERATOR_VULKAN_INSTANCE_LAYER_ENUMERATION_FILE, availableExtensions,
			availableLayers);
	}

	// ApplicationInfo
	vk::ApplicationInfo structApplicationInfo = {.sType = vk::StructureType::eApplicationInfo,
												 .pNext = std::nullptr_t(),
												 .pApplicationName = "feather-accelerator-vulkan",
												 .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
												 .pEngineName = "feather-accelerator-vulkan",
												 .engineVersion = VK_MAKE_VERSION(0, 1, 0),
												 .apiVersion = VK_API_VERSION_1_3};

	// DebugUtilsMessengerCreateInfoEXT
	vk::DebugUtilsMessengerCreateInfoEXT structDebugUtilsMessengerCreateInfo{
		.sType = vk::StructureType::eDebugUtilsMessengerCreateInfoEXT,
		.pNext = std::nullptr_t(),
		.flags = vk::DebugUtilsMessengerCreateFlagsEXT(),
		.messageSeverity = vk::DebugUtilsMessageSeverityFlagsEXT(
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eError |
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning),
		.messageType =
			vk::DebugUtilsMessageTypeFlagsEXT(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
											  vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
											  vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation),
		.pfnUserCallback = fnDebugCallback,
		.pUserData = std::nullptr_t()};

	// InstanceCreateInfo
	vk::InstanceCreateInfo structInstanceCreateInfo{
		.sType = vk::StructureType::eInstanceCreateInfo,
		.pNext = &structDebugUtilsMessengerCreateInfo,
		.flags = vk::InstanceCreateFlags(),
		.pApplicationInfo = &structApplicationInfo,
		.enabledLayerCount = (uint32_t)requiredLayers.size(),
		.ppEnabledLayerNames = requiredLayers.data(),
		.enabledExtensionCount = (uint32_t)requiredExtensions.size(),
		.ppEnabledExtensionNames = requiredExtensions.data()};

	// Create Instance
	m_instance = vk::createInstance(structInstanceCreateInfo);
	// Dynamic Loader
	m_dldi = vk::detail::DispatchLoaderDynamic(m_instance, vkGetInstanceProcAddr);
	// Create DebugUtilsMessengerEXT
	m_debugUtilsMessenger = m_instance.createDebugUtilsMessengerEXT(
		structDebugUtilsMessengerCreateInfo, std::nullptr_t(), m_dldi);
}

VKAPI_ATTR VkBool32 VKAPI_CALL InstanceManager::fnDebugCallback(
	vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	vk::DebugUtilsMessageTypeFlagsEXT messageTypes,
	vk::DebugUtilsMessengerCallbackDataEXT const *pCallbackData, void *pUserData)
{
	switch (messageSeverity) {
	case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
		std::cout << "[Feather][vk][Error]:"
				  << "\n[Type]: " << vk::to_string(messageTypes)
				  << "\n[ID]: " << pCallbackData->pMessageIdName
				  << "\n[CmdBuffer Label Count]: " << pCallbackData->cmdBufLabelCount
				  << "\n[CmdBuffer Labels]: " << pCallbackData->pCmdBufLabels
				  << "\n[Queue Label Count]: " << pCallbackData->queueLabelCount
				  << "\n[Queue Labels]: " << pCallbackData->pQueueLabels
				  << "\n[Object Count]: " << pCallbackData->objectCount
				  << "\n[Objects]: " << pCallbackData->pObjects
				  << "\n[Message]: " << pCallbackData->pMessage << std::endl;
		break;
	case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
		std::cout << "[Feather][vk][Warning]: " << pCallbackData->pMessage << std::endl;
		break;
	}
	return false;
}

} // namespace vulkan
} // namespace feather