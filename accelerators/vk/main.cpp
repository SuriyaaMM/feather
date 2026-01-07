#include "managers.hpp"
#include "foundation.hpp"
#include <exception>
#include <filesystem>

int main(int, char **)
{
	try {
		if (!std::filesystem::exists(FEATHER_ACCELERATOR_VULKAN_LOGDIR)) {
			std::filesystem::create_directories(FEATHER_ACCELERATOR_VULKAN_LOGDIR);
		}

		feather::vulkan::InstanceManager instanceManager = {};
		feather::vulkan::DeviceManager deviceManager(instanceManager.getInstanceRef());

	} catch (vk::SystemError &e) {
		FEATHER_DUMP(e.what());
	} catch (std::exception &e) {
		FEATHER_DUMP(e.what());
	}
	return 0;
}