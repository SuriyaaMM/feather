#pragma once
#include <vulkan/vulkan.hpp>

namespace feather
{
namespace vulkan
{

/*
===----- InstanceManager class -----===
# Creates Vulkan Instance, handles Debug Utils Messenger & Dynamic Dispatcher
*/
class InstanceManager
{
public:
	// Constructores, Destructors, Movers & Copiers
	InstanceManager();
	~InstanceManager() = default;
	InstanceManager(const InstanceManager &) = delete;
	InstanceManager(InstanceManager &&) = delete;
	InstanceManager operator=(const InstanceManager &) = delete;

	// Getters & Setters

	[[nodiscard]] inline vk::Instance &getInstanceRef()
	{
		return m_instance;
	}

	[[nodiscard]] inline vk::Instance *getInstancePointer()
	{
		return &m_instance;
	}

	[[nodiscard]] inline vk::detail::DispatchLoaderDynamic &getDldiRef()
	{
		return m_dldi;
	}

	[[nodiscard]] inline vk::detail::DispatchLoaderDynamic *getDldiPointer()
	{
		return &m_dldi;
	}

	// private Variables
private:
	vk::Instance m_instance;
	vk::detail::DispatchLoaderDynamic m_dldi;
	vk::DebugUtilsMessengerEXT m_debugUtilsMessenger;

	// private Functions
private:
	static VKAPI_ATTR VkBool32 VKAPI_CALL
	fnDebugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
					vk::DebugUtilsMessageTypeFlagsEXT messageTypes,
					vk::DebugUtilsMessengerCallbackDataEXT const *pCallbackData, void *pUserData);
};

/*
===----- DeviceManager class -----===
# Creates Vulkan Device (Physical & Logical)
*/
class DeviceManager
{
public:
	// Constructores, Destructors, Movers & Copiers
	DeviceManager(vk::Instance &rInstance);
	~DeviceManager() = default;
	DeviceManager(const DeviceManager &) = delete;
	DeviceManager(DeviceManager &&) = delete;
	DeviceManager operator=(const DeviceManager &) = delete;

	// Getters & Setters

private:
	vk::Instance &m_instanceRef;
	vk::PhysicalDevice m_physicalDevice;
	vk::Device m_logicalDevice;
};
} // namespace vulkan
} // namespace feather