#pragma once
#include <iostream>
#include <source_location>
#include <filesystem>
#include <fstream>

#define FEATHER_LOG(__MESSAGE) std::cout << "[Feather]: " << __MESSAGE << "\n";

#define FEATHER_DUMP(__MESSAGE)                                                                    \
	std::cout << "[Feather]: " << __MESSAGE                                                        \
			  << "\n[File]: " << std::source_location::current().file_name()                       \
			  << "\n[Line]: " << std::source_location::current().line()                            \
			  << "\n[Function]: " << std::source_location::current().function_name() << "\n";

#define FEATHER_DUMP_IF(__CONDITION, __MESSAGE)                                                    \
	if (__CONDITION) {                                                                             \
		FEATHER_DUMP(__MESSAGE);                                                                   \
	}

#define FEATHER_TRY_BLOCK(__SCOPE_OF_CODE)                                                         \
	try {                                                                                          \
		__SCOPE_OF_CODE                                                                            \
	} catch (vk::SystemError & e) {                                                                \
		FEATHER_DUMP(e.what());                                                                    \
	} catch (std::exception & e) {                                                                 \
		FEATHER_DUMP(e.what());                                                                    \
	}

/*
===----- Vulkan Dump Directory Definitions -----===
# Dumps the respective file to corresponding directory
*/
#define FEATHER_ACCELERATOR_VULKAN_LOGDIR std::filesystem::path(".feather_accelerator_vulkan_logs")

#define FEATHER_ACCELERATOR_VULKAN_INSTANCE_EXTENSION_ENUMERATION_FILE                             \
	std::filesystem::path("vulkan_instance_extensions")
#define FEATHER_ACCELERATOR_VULKAN_INSTANCE_LAYER_ENUMERATION_FILE                                 \
	std::filesystem::path("vulkan_instance_layers")

#define FEATHER_ACCELERATOR_VULKAN_DEVICE_EXTENSION_ENUMERATION_FILE(i)                            \
	std::filesystem::path("vulkan_device_extensions-" #i)
#define FEATHER_ACCELERATOR_VULKAN_DEVICE_LAYER_ENUMERATION_FILE(i)                                \
	std::filesystem::path("vulkan_device_layers-" #i)

/*
===----- FEATHER_ACCELERATOR_VULKAN_DUMP_EXTENSIONS_AND_LAYERS_TO -----===
# Does exactly what it says, Dumps them to an File
# Callee's responsibility to create the FEATHER_ACCLERATOR_VULKAN_LOGDIR directory
# Uses `std::fstream` to create write handles & also throws exceptions on bad/failure
# Creates a Temporary Scope to create handlers & write, no need of managing memory explicitly
*/
#define FEATHER_ACCELERATOR_VULKAN_DUMP_EXTENSIONS_AND_LAYERS_TO(__EXTENSION_FILE, __LAYER_FILE,   \
																 __EXTENSIONS_VAR, __LAYERS_VAR)   \
	{                                                                                              \
		/* Create Extension Paths */                                                               \
		std::filesystem::path extensionFilePath =                                                  \
			FEATHER_ACCELERATOR_VULKAN_LOGDIR / __EXTENSION_FILE;                                  \
                                                                                                   \
		std::filesystem::path layerFilePath = FEATHER_ACCELERATOR_VULKAN_LOGDIR / __LAYER_FILE;    \
                                                                                                   \
		/* Create Stream Handlers */                                                               \
		std::fstream extensionFileHandle(extensionFilePath, std::ios::out);                        \
		extensionFileHandle.exceptions(std::fstream::badbit | std::fstream::failbit);              \
		std::fstream layerFileHandle(layerFilePath, std::ios::out);                                \
		layerFileHandle.exceptions(std::fstream::badbit | std::fstream::failbit);                  \
                                                                                                   \
		/* Write */                                                                                \
		for (const auto &properties : __EXTENSIONS_VAR) {                                          \
			extensionFileHandle << "[Name]: " << properties.extensionName                          \
								<< "\n[SpecVersion]: " << properties.specVersion                   \
								<< "\n------------------------------\n";                           \
		}                                                                                          \
		extensionFileHandle.close();                                                               \
		for (const auto &properties : __LAYERS_VAR) {                                              \
			layerFileHandle << "[Name]: " << properties.layerName                                  \
							<< "\n[SpecVersion]: " << properties.specVersion                       \
							<< "\n[Description]: " << properties.description                       \
							<< "\n------------------------------\n";                               \
		}                                                                                          \
		layerFileHandle.close();                                                                   \
	}
