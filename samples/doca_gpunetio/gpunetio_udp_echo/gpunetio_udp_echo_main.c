/*
 * Copyright (c) 2023-2025 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <doca_argp.h>
#include "gpunetio_common.h"

DOCA_LOG_REGISTER(GPUNETIO_UDP_ECHO::MAIN);

static doca_error_t gpu_pci_address_callback(void *param, void *config)
{
	struct sample_echo_cfg *cfg = (struct sample_echo_cfg *)config;
	char *addr = (char *)param;
	size_t len = strnlen(addr, MAX_PCI_ADDRESS_LEN);

	if (len >= MAX_PCI_ADDRESS_LEN) {
		DOCA_LOG_ERR("GPU PCIe address too long (max %d)", MAX_PCI_ADDRESS_LEN - 1);
		return DOCA_ERROR_INVALID_VALUE;
	}
	strncpy(cfg->gpu_pcie_addr, addr, len + 1);
	return DOCA_SUCCESS;
}

static doca_error_t nic_pci_address_callback(void *param, void *config)
{
	struct sample_echo_cfg *cfg = (struct sample_echo_cfg *)config;
	char *addr = (char *)param;
	size_t len = strnlen(addr, MAX_PCI_ADDRESS_LEN);

	if (len >= MAX_PCI_ADDRESS_LEN) {
		DOCA_LOG_ERR("NIC PCIe address too long (max %d)", MAX_PCI_ADDRESS_LEN - 1);
		return DOCA_ERROR_INVALID_VALUE;
	}
	strncpy(cfg->nic_pcie_addr, addr, len + 1);
	return DOCA_SUCCESS;
}

static doca_error_t proxy_callback(void *param, void *config)
{
	struct sample_echo_cfg *cfg = (struct sample_echo_cfg *)config;

	cfg->cpu_proxy = (*(uint32_t *)param != 0);
	return DOCA_SUCCESS;
}

static doca_error_t register_sample_params(void)
{
	doca_error_t result;
	struct doca_argp_param *gpu_param, *nic_param, *proxy_param;

	/* -g / --gpu */
	result = doca_argp_param_create(&gpu_param);
	if (result != DOCA_SUCCESS)
		return result;
	doca_argp_param_set_short_name(gpu_param, "g");
	doca_argp_param_set_long_name(gpu_param, "gpu");
	doca_argp_param_set_arguments(gpu_param, "<GPU PCIe address>");
	doca_argp_param_set_description(gpu_param, "GPU PCIe address (e.g. c2:00.0)");
	doca_argp_param_set_callback(gpu_param, gpu_pci_address_callback);
	doca_argp_param_set_type(gpu_param, DOCA_ARGP_TYPE_STRING);
	doca_argp_param_set_mandatory(gpu_param);
	result = doca_argp_register_param(gpu_param);
	if (result != DOCA_SUCCESS)
		return result;

	/* -n / --nic */
	result = doca_argp_param_create(&nic_param);
	if (result != DOCA_SUCCESS)
		return result;
	doca_argp_param_set_short_name(nic_param, "n");
	doca_argp_param_set_long_name(nic_param, "nic");
	doca_argp_param_set_arguments(nic_param, "<NIC PCIe address>");
	doca_argp_param_set_description(nic_param, "NIC PCIe address (e.g. 41:00.0)");
	doca_argp_param_set_callback(nic_param, nic_pci_address_callback);
	doca_argp_param_set_type(nic_param, DOCA_ARGP_TYPE_STRING);
	doca_argp_param_set_mandatory(nic_param);
	result = doca_argp_register_param(nic_param);
	if (result != DOCA_SUCCESS)
		return result;

	/* -p / --proxy */
	result = doca_argp_param_create(&proxy_param);
	if (result != DOCA_SUCCESS)
		return result;
	doca_argp_param_set_short_name(proxy_param, "p");
	doca_argp_param_set_long_name(proxy_param, "proxy");
	doca_argp_param_set_arguments(proxy_param, "<0|1>");
	doca_argp_param_set_description(proxy_param,
					"CPU proxy mode: 0=off, 1=on. Use 1 when GPU and NIC are on "
					"different PCIe roots (NODE topology, no GDRCopy).");
	doca_argp_param_set_callback(proxy_param, proxy_callback);
	doca_argp_param_set_type(proxy_param, DOCA_ARGP_TYPE_INT);
	result = doca_argp_register_param(proxy_param);
	if (result != DOCA_SUCCESS)
		return result;

	return DOCA_SUCCESS;
}

int main(int argc, char **argv)
{
	doca_error_t result;
	struct doca_log_backend *sdk_log;
	struct sample_echo_cfg sample_cfg = {0};
	int exit_status = EXIT_FAILURE;
	cudaError_t cuda_ret;

	result = doca_log_backend_create_standard();
	if (result != DOCA_SUCCESS)
		goto sample_exit;

	result = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
	if (result != DOCA_SUCCESS)
		goto sample_exit;
	result = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);
	if (result != DOCA_SUCCESS)
		goto sample_exit;

	DOCA_LOG_INFO("Starting GPUNetIO UDP echo + latency sample");

	sample_cfg.cpu_proxy = false;

	result = doca_argp_init(NULL, &sample_cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init ARGP: %s", doca_error_get_descr(result));
		goto sample_exit;
	}

	result = register_sample_params();
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register params: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}

	result = doca_argp_start(argc, argv);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse args: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}

	/* Trigger CUDA runtime init */
	cuda_ret = cudaFree(0);
	if (cuda_ret != cudaSuccess) {
		DOCA_LOG_ERR("CUDA init failed: %s", cudaGetErrorString(cuda_ret));
		goto argp_cleanup;
	}

	cuda_ret = cudaDeviceGetByPCIBusId(&sample_cfg.cuda_id, sample_cfg.gpu_pcie_addr);
	if (cuda_ret != cudaSuccess) {
		DOCA_LOG_ERR("Invalid GPU PCIe address %s: %s",
			     sample_cfg.gpu_pcie_addr, cudaGetErrorString(cuda_ret));
		goto argp_cleanup;
	}
	cudaSetDevice(sample_cfg.cuda_id);

	DOCA_LOG_INFO("Config: GPU=%s  NIC=%s  cpu_proxy=%s",
		      sample_cfg.gpu_pcie_addr,
		      sample_cfg.nic_pcie_addr,
		      sample_cfg.cpu_proxy ? "on" : "off");

	result = gpunetio_udp_echo(&sample_cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("gpunetio_udp_echo failed: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}

	exit_status = EXIT_SUCCESS;

argp_cleanup:
	doca_argp_destroy();
sample_exit:
	DOCA_LOG_INFO("Sample %s", (exit_status == EXIT_SUCCESS) ? "finished successfully" : "finished with errors");
	return exit_status;
}
