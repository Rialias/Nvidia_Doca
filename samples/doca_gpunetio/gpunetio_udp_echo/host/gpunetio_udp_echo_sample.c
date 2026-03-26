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

/*
 * gpunetio_udp_echo_sample.c
 *
 * Host-side code for the GPUNetIO UDP echo + latency sample.
 *
 * Topology
 * --------
 *   NIC (41:00.0) ──RXQ──> GPU (c2:00.0) kernel ──TXQ──> NIC (41:00.0)
 *
 * The GPU CUDA kernel (in device/gpunetio_udp_echo_kernel.cu) waits for
 * incoming UDP packets on the RXQ, copies them to a TX buffer, swaps
 * MACs/IPs/ports, and retransmits on the same port.  It records a latency
 * sample (NIC HW RX timestamp → GPU timer just before txq_send) each batch.
 *
 * CPU proxy mode
 * --------------
 * When --proxy 1 is given, both the RXQ and TXQ doorbells are placed on
 * CPU-accessible memory (via doca_eth_rxq_gpu_set_uar_on_cpu /
 * doca_eth_txq_gpu_set_uar_on_cpu).  Two CPU threads (rxq_proxy, txq_proxy)
 * call the respective progress functions in a tight loop, ringing the NIC
 * UAR from the CPU.  This is needed when GPU and NIC are on different PCIe
 * roots (NODE topology) where GPU-side MMIO to the NIC is not supported.
 */

#include <arpa/inet.h>
#include <doca_flow.h>
#include "gpunetio_common.h"
#include "common.h"

#define FLOW_NB_COUNTERS 524228 /* 1024 × 512 */

/* DOCA Flow port handle – module-level so helper functions can reach it */
static struct doca_flow_port *df_port;

/* Set to true by the SIGINT/SIGTERM handler */
bool force_quit;

DOCA_LOG_REGISTER(GPUNETIO_UDP_ECHO);

/* ------------------------------------------------------------------ */
/* Utilities                                                           */
/* ------------------------------------------------------------------ */

static size_t get_host_page_size(void)
{
	long ret = sysconf(_SC_PAGESIZE);
	return (ret == -1) ? 4096 : (size_t)ret;
}

static void signal_handler(int signum)
{
	if (signum == SIGINT || signum == SIGTERM) {
		DOCA_LOG_INFO("Signal %d received, preparing to exit!", signum);
		DOCA_GPUNETIO_VOLATILE(force_quit) = true;
	}
}

/* ------------------------------------------------------------------ */
/* DOCA device                                                         */
/* ------------------------------------------------------------------ */

static doca_error_t init_doca_device(char *nic_pcie_addr, struct doca_dev **ddev)
{
	if (nic_pcie_addr == NULL || ddev == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	if (strnlen(nic_pcie_addr, DOCA_DEVINFO_PCI_ADDR_SIZE) >= DOCA_DEVINFO_PCI_ADDR_SIZE)
		return DOCA_ERROR_INVALID_VALUE;

	return open_doca_device_with_pci(nic_pcie_addr, NULL, ddev);
}

/* ------------------------------------------------------------------ */
/* DOCA Flow                                                           */
/* ------------------------------------------------------------------ */

static doca_error_t init_doca_flow(void)
{
	struct doca_flow_cfg *flow_cfg;
	doca_error_t result;

	result = doca_flow_cfg_create(&flow_cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create doca_flow_cfg: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_cfg_set_pipe_queues(flow_cfg, 1);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set pipe_queues: %s", doca_error_get_descr(result));
		doca_flow_cfg_destroy(flow_cfg);
		return result;
	}

	result = doca_flow_cfg_set_mode_args(flow_cfg, "vnf,isolated");
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set mode_args: %s", doca_error_get_descr(result));
		doca_flow_cfg_destroy(flow_cfg);
		return result;
	}

	result = doca_flow_cfg_set_nr_counters(flow_cfg, FLOW_NB_COUNTERS);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set nr_counters: %s", doca_error_get_descr(result));
		doca_flow_cfg_destroy(flow_cfg);
		return result;
	}

	result = doca_flow_init(flow_cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init DOCA flow: %s", doca_error_get_descr(result));
		doca_flow_cfg_destroy(flow_cfg);
		return result;
	}

	doca_flow_cfg_destroy(flow_cfg);
	return DOCA_SUCCESS;
}

static doca_error_t start_doca_flow(struct doca_dev *dev)
{
	struct doca_flow_port_cfg *port_cfg;
	doca_error_t result;

	result = doca_flow_port_cfg_create(&port_cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create flow port cfg: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_port_cfg_set_port_id(port_cfg, 0);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set port ID: %s", doca_error_get_descr(result));
		doca_flow_port_cfg_destroy(port_cfg);
		return result;
	}

	result = doca_flow_port_cfg_set_dev(port_cfg, dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set port dev: %s", doca_error_get_descr(result));
		doca_flow_port_cfg_destroy(port_cfg);
		return result;
	}

	result = doca_flow_port_start(port_cfg, &df_port);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to start flow port: %s", doca_error_get_descr(result));
		doca_flow_port_cfg_destroy(port_cfg);
		return result;
	}

	doca_flow_port_cfg_destroy(port_cfg);
	return DOCA_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* DOCA Flow pipes (UDP steering → GPU RXQ)                           */
/* ------------------------------------------------------------------ */

/*
 * Create "GPU_RXQ_UDP_PIPE" — matches all IPv4/UDP, steers to queue 0.
 * That queue is the echo RXQ.
 */
static doca_error_t create_udp_pipe(struct echo_rxq *rxq)
{
	doca_error_t result;
	struct doca_flow_match match = {0};
	struct doca_flow_fwd fwd = {0};
	struct doca_flow_fwd miss_fwd = {0};
	struct doca_flow_pipe_cfg *pipe_cfg;
	struct doca_flow_pipe_entry *entry;
	uint16_t rss_queues[1];
	struct doca_flow_monitor monitor = {
		.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED,
	};
	const char *pipe_name = "GPU_RXQ_UDP_PIPE";

	if (rxq == NULL || df_port == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	match.parser_meta.outer_l3_type = DOCA_FLOW_L3_META_IPV4;
	match.parser_meta.outer_l4_type = DOCA_FLOW_L4_META_UDP;

	doca_eth_rxq_apply_queue_id(rxq->eth_rxq_cpu, 0);
	rss_queues[0] = 0;

	fwd.type               = DOCA_FLOW_FWD_RSS;
	fwd.rss_type           = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;
	fwd.rss.queues_array   = rss_queues;
	fwd.rss.outer_flags    = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_UDP;
	fwd.rss.nr_queues      = 1;
	miss_fwd.type          = DOCA_FLOW_FWD_DROP;

	result = doca_flow_pipe_cfg_create(&pipe_cfg, df_port);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create pipe cfg: %s", doca_error_get_descr(result));
		return result;
	}

	if ((result = doca_flow_pipe_cfg_set_name(pipe_cfg, pipe_name)) != DOCA_SUCCESS ||
	    (result = doca_flow_pipe_cfg_set_type(pipe_cfg, DOCA_FLOW_PIPE_BASIC)) != DOCA_SUCCESS ||
	    (result = doca_flow_pipe_cfg_set_is_root(pipe_cfg, false)) != DOCA_SUCCESS ||
	    (result = doca_flow_pipe_cfg_set_match(pipe_cfg, &match, NULL)) != DOCA_SUCCESS ||
	    (result = doca_flow_pipe_cfg_set_monitor(pipe_cfg, &monitor)) != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to configure UDP pipe: %s", doca_error_get_descr(result));
		doca_flow_pipe_cfg_destroy(pipe_cfg);
		return result;
	}

	result = doca_flow_pipe_create(pipe_cfg, &fwd, &miss_fwd, &rxq->rxq_pipe);
	doca_flow_pipe_cfg_destroy(pipe_cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("UDP pipe creation failed: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_pipe_add_entry(0, rxq->rxq_pipe, &match, 0, NULL, NULL, NULL,
					  DOCA_FLOW_NO_WAIT, NULL, &entry);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("UDP pipe entry creation failed: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_entries_process(df_port, 0, 0, 0);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("UDP pipe entry process failed: %s", doca_error_get_descr(result));
		return result;
	}

	DOCA_LOG_DBG("Created pipe %s", pipe_name);
	return DOCA_SUCCESS;
}

/* Root control pipe: forwards IPv4/UDP to rxq_pipe, drops everything else */
static doca_error_t create_root_pipe(struct echo_rxq *rxq)
{
	doca_error_t result;
	struct doca_flow_monitor monitor = {
		.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED,
	};
	struct doca_flow_match udp_match = {
		.outer.eth.type    = htons(DOCA_FLOW_ETHER_TYPE_IPV4),
		.outer.l3_type     = DOCA_FLOW_L3_TYPE_IP4,
		.outer.ip4.next_proto = IPPROTO_UDP,
	};
	struct doca_flow_fwd udp_fwd = {
		.type      = DOCA_FLOW_FWD_PIPE,
		.next_pipe = rxq->rxq_pipe,
	};
	struct doca_flow_pipe_cfg *pipe_cfg;
	const char *pipe_name = "ROOT_PIPE";

	if (rxq == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	result = doca_flow_pipe_cfg_create(&pipe_cfg, df_port);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create root pipe cfg: %s", doca_error_get_descr(result));
		return result;
	}

	if ((result = doca_flow_pipe_cfg_set_name(pipe_cfg, pipe_name)) != DOCA_SUCCESS ||
	    (result = doca_flow_pipe_cfg_set_type(pipe_cfg, DOCA_FLOW_PIPE_CONTROL)) != DOCA_SUCCESS ||
	    (result = doca_flow_pipe_cfg_set_is_root(pipe_cfg, true)) != DOCA_SUCCESS ||
	    (result = doca_flow_pipe_cfg_set_monitor(pipe_cfg, &monitor)) != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to configure root pipe: %s", doca_error_get_descr(result));
		doca_flow_pipe_cfg_destroy(pipe_cfg);
		return result;
	}

	result = doca_flow_pipe_create(pipe_cfg, NULL, NULL, &rxq->root_pipe);
	doca_flow_pipe_cfg_destroy(pipe_cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Root pipe creation failed: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_pipe_control_add_entry(0, 0, rxq->root_pipe,
						  &udp_match, NULL, NULL, NULL, NULL, NULL,
						  NULL, &udp_fwd, NULL, &rxq->root_udp_entry);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Root pipe UDP entry creation failed: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_entries_process(df_port, 0, 0, 0);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Root pipe entry process failed: %s", doca_error_get_descr(result));
		return result;
	}

	DOCA_LOG_DBG("Created pipe %s", pipe_name);
	return DOCA_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* RXQ                                                                 */
/* ------------------------------------------------------------------ */

static doca_error_t destroy_rxq(struct echo_rxq *rxq)
{
	doca_error_t result;

	if (rxq == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	DOCA_LOG_INFO("Destroying RXQ");

	if (rxq->root_pipe != NULL)
		doca_flow_pipe_destroy(rxq->root_pipe);

	if (rxq->rxq_pipe != NULL)
		doca_flow_pipe_destroy(rxq->rxq_pipe);

	if (df_port != NULL) {
		result = doca_flow_port_stop(df_port);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to stop DOCA flow port: %s", doca_error_get_name(result));
			return DOCA_ERROR_BAD_STATE;
		}
	}

	if (rxq->eth_rxq_ctx != NULL) {
		result = doca_ctx_stop(rxq->eth_rxq_ctx);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed doca_ctx_stop: %s", doca_error_get_descr(result));
			return DOCA_ERROR_BAD_STATE;
		}
	}

	if (rxq->gpu_pkt_addr != NULL) {
		result = doca_gpu_mem_free(rxq->gpu_dev, rxq->gpu_pkt_addr);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to free GPU RXQ buf: %s", doca_error_get_descr(result));
			return DOCA_ERROR_BAD_STATE;
		}
	}

	if (rxq->eth_rxq_cpu != NULL) {
		result = doca_eth_rxq_destroy(rxq->eth_rxq_cpu);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed doca_eth_rxq_destroy: %s", doca_error_get_descr(result));
			return DOCA_ERROR_BAD_STATE;
		}
	}

	if (rxq->pkt_buff_mmap != NULL) {
		result = doca_mmap_destroy(rxq->pkt_buff_mmap);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to destroy RXQ mmap: %s", doca_error_get_descr(result));
			return DOCA_ERROR_BAD_STATE;
		}
	}

	if (rxq->ddev != NULL) {
		result = doca_dev_close(rxq->ddev);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to close ddev: %s", doca_error_get_descr(result));
			return DOCA_ERROR_BAD_STATE;
		}
	}

	if (df_port != NULL)
		doca_flow_destroy();

	return DOCA_SUCCESS;
}

/*
 * create_echo_rxq — allocates and starts a GPU-bound cyclic RX queue.
 *
 * Mirrors create_rxq() from gpunetio_simple_receive with the same
 * MCST-disable fix for pre-Hopper GPUs in cpu_proxy mode.
 */
static doca_error_t create_echo_rxq(struct echo_rxq *rxq,
				    struct doca_gpu *gpu_dev,
				    int cuda_id,
				    struct doca_dev *ddev,
				    bool cpu_proxy)
{
	doca_error_t result;
	uint32_t cyclic_buffer_size = 0;
	struct cudaDeviceProp prop;

	if (rxq == NULL || gpu_dev == NULL || ddev == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	rxq->gpu_dev = gpu_dev;
	rxq->ddev    = ddev;
	rxq->port    = df_port;

	DOCA_LOG_INFO("Creating echo RXQ");

	result = doca_eth_rxq_create(rxq->ddev, MAX_PKT_NUM, MAX_PKT_SIZE, &rxq->eth_rxq_cpu);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed doca_eth_rxq_create: %s", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_eth_rxq_set_type(rxq->eth_rxq_cpu, DOCA_ETH_RXQ_TYPE_CYCLIC);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed doca_eth_rxq_set_type: %s", doca_error_get_descr(result));
		goto exit_error;
	}

	result = doca_eth_rxq_estimate_packet_buf_size(DOCA_ETH_RXQ_TYPE_CYCLIC,
						       0, 0, MAX_PKT_SIZE, MAX_PKT_NUM,
						       0, 0, 0,
						       &cyclic_buffer_size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to estimate cyclic buffer size: %s", doca_error_get_descr(result));
		goto exit_error;
	}

	result = doca_mmap_create(&rxq->pkt_buff_mmap);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create RXQ mmap: %s", doca_error_get_descr(result));
		goto exit_error;
	}

	result = doca_mmap_add_dev(rxq->pkt_buff_mmap, rxq->ddev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add dev to RXQ mmap: %s", doca_error_get_descr(result));
		goto exit_error;
	}

	ALIGN_SIZE(cyclic_buffer_size, get_host_page_size());

	result = doca_gpu_mem_alloc(rxq->gpu_dev,
				    cyclic_buffer_size,
				    get_host_page_size(),
				    DOCA_GPU_MEM_TYPE_GPU,
				    (void **)&rxq->gpu_pkt_addr,
				    NULL);
	if (result != DOCA_SUCCESS || rxq->gpu_pkt_addr == NULL) {
		DOCA_LOG_ERR("Failed to allocate GPU RXQ memory: %s", doca_error_get_descr(result));
		goto exit_error;
	}

	/* Try dmabuf mapping first; fall back to nvidia-peermem */
	result = doca_gpu_dmabuf_fd(rxq->gpu_dev, rxq->gpu_pkt_addr,
				    cyclic_buffer_size, &rxq->dmabuf_fd);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_INFO("RXQ buffer (0x%p %u B): nvidia-peermem mode",
			      rxq->gpu_pkt_addr, cyclic_buffer_size);
		result = doca_mmap_set_memrange(rxq->pkt_buff_mmap,
						rxq->gpu_pkt_addr, cyclic_buffer_size);
	} else {
		DOCA_LOG_INFO("RXQ buffer (0x%p %u B fd %d): dmabuf mode",
			      rxq->gpu_pkt_addr, cyclic_buffer_size, rxq->dmabuf_fd);
		result = doca_mmap_set_dmabuf_memrange(rxq->pkt_buff_mmap,
						       rxq->dmabuf_fd,
						       rxq->gpu_pkt_addr,
						       0,
						       cyclic_buffer_size);
	}
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set RXQ mmap memrange: %s", doca_error_get_descr(result));
		goto exit_error;
	}

	result = doca_mmap_set_permissions(rxq->pkt_buff_mmap, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set RXQ mmap permissions: %s", doca_error_get_descr(result));
		goto exit_error;
	}

	result = doca_mmap_start(rxq->pkt_buff_mmap);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to start RXQ mmap: %s", doca_error_get_descr(result));
		goto exit_error;
	}

	/*
	 * Obtain an NIC mkey for the RX cyclic buffer so txq_send can DMA-read
	 * directly from it (zero-copy echo: no separate TX buffer needed).
	 */
	result = doca_mmap_get_mkey(rxq->pkt_buff_mmap, rxq->ddev, &rxq->rx_pkt_mkey);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to get RXQ mmap mkey: %s", doca_error_get_descr(result));
		goto exit_error;
	}
	rxq->rx_pkt_mkey = htobe32(rxq->rx_pkt_mkey);

	result = doca_eth_rxq_set_pkt_buf(rxq->eth_rxq_cpu, rxq->pkt_buff_mmap,
					  0, cyclic_buffer_size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set RXQ cyclic buffer: %s", doca_error_get_descr(result));
		goto exit_error;
	}

	/*
	 * On pre-Hopper (major < 9) GPUs, the MCST dump QP provides a PCIe-read
	 * flush to ensure packet DMA data is coherent for the GPU SM.
	 * In cpu_proxy mode the MCST QP's UAR is on the GPU side (MMIO), which
	 * the GPU cannot ring across a NODE PCIe link — it causes kernel hangs.
	 * So: enable MCST only on pre-Hopper AND NOT in cpu_proxy mode.
	 */
	cudaGetDeviceProperties(&prop, cuda_id);
	if (prop.major < 9 && !cpu_proxy) {
		result = doca_eth_rxq_gpu_enable_mcst_qp(rxq->eth_rxq_cpu);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to enable MCST QP: %s", doca_error_get_descr(result));
			goto exit_error;
		}
	}

	if (cpu_proxy) {
		result = doca_eth_rxq_gpu_set_uar_on_cpu(rxq->eth_rxq_cpu);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed doca_eth_rxq_gpu_set_uar_on_cpu: %s",
				     doca_error_get_descr(result));
			goto exit_error;
		}
	}

	rxq->eth_rxq_ctx = doca_eth_rxq_as_doca_ctx(rxq->eth_rxq_cpu);
	if (rxq->eth_rxq_ctx == NULL) {
		DOCA_LOG_ERR("Failed doca_eth_rxq_as_doca_ctx");
		goto exit_error;
	}

	result = doca_ctx_set_datapath_on_gpu(rxq->eth_rxq_ctx, rxq->gpu_dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed doca_ctx_set_datapath_on_gpu (RXQ): %s",
			     doca_error_get_descr(result));
		goto exit_error;
	}

	result = doca_ctx_start(rxq->eth_rxq_ctx);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed doca_ctx_start (RXQ): %s", doca_error_get_descr(result));
		goto exit_error;
	}

	result = doca_eth_rxq_get_gpu_handle(rxq->eth_rxq_cpu, &rxq->eth_rxq_gpu);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed doca_eth_rxq_get_gpu_handle: %s", doca_error_get_descr(result));
		goto exit_error;
	}

	result = create_udp_pipe(rxq);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("create_udp_pipe failed: %s", doca_error_get_descr(result));
		goto exit_error;
	}

	result = create_root_pipe(rxq);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("create_root_pipe failed: %s", doca_error_get_descr(result));
		goto exit_error;
	}

	return DOCA_SUCCESS;

exit_error:
	destroy_rxq(rxq);
	return DOCA_ERROR_BAD_STATE;
}

/* ------------------------------------------------------------------ */
/* TXQ                                                                 */
/* ------------------------------------------------------------------ */

static doca_error_t destroy_txq(struct echo_txq *txq)
{
	doca_error_t result;

	if (txq == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	DOCA_LOG_INFO("Destroying TXQ");

	if (txq->eth_txq_ctx != NULL) {
		result = doca_ctx_stop(txq->eth_txq_ctx);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to stop TXQ ctx: %s", doca_error_get_descr(result));
			return DOCA_ERROR_BAD_STATE;
		}
	}

	if (txq->eth_txq_cpu != NULL) {
		result = doca_eth_txq_destroy(txq->eth_txq_cpu);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed doca_eth_txq_destroy: %s", doca_error_get_descr(result));
			return DOCA_ERROR_BAD_STATE;
		}
	}

	return DOCA_SUCCESS;
}

/*
 * create_echo_txq — allocates and starts a GPU-bound transmit queue.
 * The echo kernel sends directly from the RX cyclic buffer (zero-copy),
 * so no separate TX data buffer is allocated here.
 */
static doca_error_t create_echo_txq(struct echo_txq *txq,
				    struct doca_gpu *gpu_dev,
				    struct doca_dev *ddev,
				    bool cpu_proxy)
{
	doca_error_t result;

	if (txq == NULL || gpu_dev == NULL || ddev == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	txq->gpu_dev = gpu_dev;
	txq->ddev    = ddev;

	DOCA_LOG_INFO("Creating echo TXQ");

	result = doca_eth_txq_create(txq->ddev, MAX_SQ_DESCR_NUM, &txq->eth_txq_cpu);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed doca_eth_txq_create: %s", doca_error_get_descr(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = doca_eth_txq_set_l3_chksum_offload(txq->eth_txq_cpu, 1);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set L3 checksum offload: %s", doca_error_get_descr(result));
		goto exit_error;
	}

	result = doca_eth_txq_set_l4_chksum_offload(txq->eth_txq_cpu, 1);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set L4 checksum offload: %s", doca_error_get_descr(result));
		goto exit_error;
	}

	/* CQ is polled by the GPU kernel */
	result = doca_eth_txq_gpu_set_completion_on_gpu(txq->eth_txq_cpu);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed doca_eth_txq_gpu_set_completion_on_gpu: %s",
			     doca_error_get_descr(result));
		goto exit_error;
	}

	if (cpu_proxy) {
		result = doca_eth_txq_gpu_set_uar_on_cpu(txq->eth_txq_cpu);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed doca_eth_txq_gpu_set_uar_on_cpu: %s",
				     doca_error_get_descr(result));
			goto exit_error;
		}
	}

	txq->eth_txq_ctx = doca_eth_txq_as_doca_ctx(txq->eth_txq_cpu);
	if (txq->eth_txq_ctx == NULL) {
		DOCA_LOG_ERR("Failed doca_eth_txq_as_doca_ctx");
		goto exit_error;
	}

	result = doca_ctx_set_datapath_on_gpu(txq->eth_txq_ctx, txq->gpu_dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed doca_ctx_set_datapath_on_gpu (TXQ): %s",
			     doca_error_get_descr(result));
		goto exit_error;
	}

	result = doca_ctx_start(txq->eth_txq_ctx);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed doca_ctx_start (TXQ): %s", doca_error_get_descr(result));
		goto exit_error;
	}

	result = doca_eth_txq_apply_queue_id(txq->eth_txq_cpu, 0);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed doca_eth_txq_apply_queue_id: %s", doca_error_get_descr(result));
		goto exit_error;
	}

	result = doca_eth_txq_get_gpu_handle(txq->eth_txq_cpu, &txq->eth_txq_gpu);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed doca_eth_txq_get_gpu_handle: %s", doca_error_get_descr(result));
		goto exit_error;
	}

	return DOCA_SUCCESS;

exit_error:
	destroy_txq(txq);
	return DOCA_ERROR_BAD_STATE;
}

/* ------------------------------------------------------------------ */
/* CPU proxy threads                                                   */
/* ------------------------------------------------------------------ */

/* Drives the RXQ doorbell from the CPU (replaces GPU UAR ring) */
static void *progress_rxq_proxy(void *args_)
{
	struct rxq_cpu_proxy_args *args = (struct rxq_cpu_proxy_args *)args_;

	printf("[RXQ PROXY] thread running\n");
	fflush(stdout);
	while (ACCESS_ONCE_64b(*args->exit_flag) == 0)
		doca_eth_rxq_gpu_cpu_proxy_progress(args->rxq);
	return NULL;
}

/* Drives the TXQ doorbell from the CPU (replaces GPU UAR ring) */
static void *progress_txq_proxy(void *args_)
{
	struct txq_cpu_proxy_args *args = (struct txq_cpu_proxy_args *)args_;

	printf("[TXQ PROXY] thread running\n");
	fflush(stdout);
	while (ACCESS_ONCE_64b(*args->exit_flag) == 0)
		doca_eth_txq_gpu_cpu_proxy_progress(args->txq);
	return NULL;
}

/* ------------------------------------------------------------------ */
/* Latency statistics (read from CPU_GPU shared ring after kernel exit)*/
/* ------------------------------------------------------------------ */

static void print_latency_stats(const struct latency_sample *buf, uint64_t count)
{
	uint64_t n = (count < LATENCY_RING_SIZE) ? count : LATENCY_RING_SIZE;
	uint64_t min_ns = UINT64_MAX, max_ns = 0, sum_ns = 0;

	if (n == 0) {
		printf("No latency samples recorded.\n");
		return;
	}

	for (uint64_t i = 0; i < n; i++) {
		uint64_t lat = buf[i].latency_ns;
		if (lat < min_ns)
			min_ns = lat;
		if (lat > max_ns)
			max_ns = lat;
		sum_ns += lat;
	}

	printf("\n=== GPU Echo Processing Latency (rxq_recv wakeup → txq_send WQE post) ===\n");
	printf("  Samples : %lu (ring size %d)\n", (unsigned long)n, LATENCY_RING_SIZE);
	printf("  Min     : %lu ns (%.2f µs)\n", (unsigned long)min_ns, min_ns / 1000.0);
	printf("  Max     : %lu ns (%.2f µs)\n", (unsigned long)max_ns, max_ns / 1000.0);
	printf("  Avg     : %.1f ns (%.2f µs)\n",
	       (double)sum_ns / (double)n, (double)sum_ns / (double)n / 1000.0);
	printf("=========================================================\n");
	fflush(stdout);
}

/* ------------------------------------------------------------------ */
/* Main sample entry point                                             */
/* ------------------------------------------------------------------ */

/*
 * gpunetio_udp_echo — set up the echo pipeline and run until Ctrl-C.
 *
 * @sample_cfg [in]: Configuration from command-line arguments
 * @return: DOCA_SUCCESS on success
 */
doca_error_t gpunetio_udp_echo(struct sample_echo_cfg *sample_cfg)
{
	doca_error_t result;
	struct doca_gpu *gpu_dev = NULL;
	struct doca_dev *ddev    = NULL;
	struct echo_rxq  rxq     = {0};
	struct echo_txq  txq     = {0};
	cudaStream_t     stream;
	cudaError_t      res_rt  = cudaSuccess;

	/* GPU-CPU shared memory: GPU reads (exit flag), CPU writes */
	uint32_t *gpu_exit_condition;
	uint32_t *cpu_exit_condition;

	/* CPU-GPU shared memory: GPU writes (latency ring/count), CPU reads */
	struct latency_sample *gpu_latency_buf;
	struct latency_sample *cpu_latency_buf;
	uint64_t *gpu_latency_count;
	uint64_t *cpu_latency_count;

	/* CPU proxy thread handles */
	struct rxq_cpu_proxy_args rxq_proxy_args = {0};
	struct txq_cpu_proxy_args txq_proxy_args = {0};
	pthread_t rxq_proxy_tid = 0, txq_proxy_tid = 0;
	bool rxq_proxy_started = false, txq_proxy_started = false;

	/* ---- DOCA device ---- */
	result = init_doca_device(sample_cfg->nic_pcie_addr, &ddev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("init_doca_device: %s", doca_error_get_descr(result));
		return result;
	}

	/* ---- DOCA Flow ---- */
	result = init_doca_flow();
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("init_doca_flow: %s", doca_error_get_descr(result));
		goto exit;
	}

	result = start_doca_flow(ddev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("start_doca_flow: %s", doca_error_get_descr(result));
		goto exit;
	}

	/* ---- Signals ---- */
	DOCA_GPUNETIO_VOLATILE(force_quit) = false;
	signal(SIGINT,  signal_handler);
	signal(SIGTERM, signal_handler);

	/* ---- GPU device ---- */
	result = doca_gpu_create(sample_cfg->gpu_pcie_addr, &gpu_dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("doca_gpu_create: %s", doca_error_get_descr(result));
		goto exit;
	}

	/* ---- Queues ---- */
	result = create_echo_rxq(&rxq, gpu_dev, sample_cfg->cuda_id, ddev, sample_cfg->cpu_proxy);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("create_echo_rxq: %s", doca_error_get_descr(result));
		goto exit;
	}

	result = create_echo_txq(&txq, gpu_dev, ddev, sample_cfg->cpu_proxy);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("create_echo_txq: %s", doca_error_get_descr(result));
		goto exit;
	}

	/* ---- CUDA stream ---- */
	res_rt = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	if (res_rt != cudaSuccess) {
		DOCA_LOG_ERR("cudaStreamCreateWithFlags: %s", cudaGetErrorString(res_rt));
		result = DOCA_ERROR_DRIVER;
		goto exit;
	}

	/* ---- Exit flag: GPU_CPU (GPU reads, CPU writes) ---- */
	result = doca_gpu_mem_alloc(gpu_dev,
				    sizeof(uint32_t),
				    get_host_page_size(),
				    DOCA_GPU_MEM_TYPE_GPU_CPU,
				    (void **)&gpu_exit_condition,
				    (void **)&cpu_exit_condition);
	if (result != DOCA_SUCCESS || gpu_exit_condition == NULL || cpu_exit_condition == NULL) {
		DOCA_LOG_ERR("Failed to alloc exit flag: %s", doca_error_get_descr(result));
		goto exit;
	}
	cpu_exit_condition[0] = 0;

	/* ---- Latency ring: CPU_GPU (GPU writes, CPU reads) ---- */
	result = doca_gpu_mem_alloc(gpu_dev,
				    LATENCY_RING_SIZE * sizeof(struct latency_sample),
				    get_host_page_size(),
				    DOCA_GPU_MEM_TYPE_CPU_GPU,
				    (void **)&gpu_latency_buf,
				    (void **)&cpu_latency_buf);
	if (result != DOCA_SUCCESS || gpu_latency_buf == NULL || cpu_latency_buf == NULL) {
		DOCA_LOG_ERR("Failed to alloc latency ring: %s", doca_error_get_descr(result));
		goto exit;
	}

	result = doca_gpu_mem_alloc(gpu_dev,
				    sizeof(uint64_t),
				    get_host_page_size(),
				    DOCA_GPU_MEM_TYPE_CPU_GPU,
				    (void **)&gpu_latency_count,
				    (void **)&cpu_latency_count);
	if (result != DOCA_SUCCESS || gpu_latency_count == NULL || cpu_latency_count == NULL) {
		DOCA_LOG_ERR("Failed to alloc latency counter: %s", doca_error_get_descr(result));
		goto exit;
	}
	cpu_latency_count[0] = 0;

	/* ---- CPU proxy threads (one for RXQ, one for TXQ) ---- */
	if (sample_cfg->cpu_proxy) {
		rxq_proxy_args.rxq       = rxq.eth_rxq_cpu;
		rxq_proxy_args.exit_flag = (uint64_t *)calloc(1, sizeof(uint64_t));
		if (rxq_proxy_args.exit_flag == NULL) {
			DOCA_LOG_ERR("Failed to allocate RXQ proxy exit flag");
			goto exit;
		}
		*rxq_proxy_args.exit_flag = 0;

		txq_proxy_args.txq       = txq.eth_txq_cpu;
		txq_proxy_args.exit_flag = (uint64_t *)calloc(1, sizeof(uint64_t));
		if (txq_proxy_args.exit_flag == NULL) {
			DOCA_LOG_ERR("Failed to allocate TXQ proxy exit flag");
			free(rxq_proxy_args.exit_flag);
			goto exit;
		}
		*txq_proxy_args.exit_flag = 0;

		if (pthread_create(&rxq_proxy_tid, NULL, progress_rxq_proxy, &rxq_proxy_args) != 0) {
			perror("Failed to create RXQ proxy thread");
			goto exit;
		}
		rxq_proxy_started = true;

		if (pthread_create(&txq_proxy_tid, NULL, progress_txq_proxy, &txq_proxy_args) != 0) {
			perror("Failed to create TXQ proxy thread");
			goto exit;
		}
		txq_proxy_started = true;
	}

	/* ---- Launch echo kernel ---- */
	DOCA_LOG_INFO("Launching GPU echo kernel (ctrl+c to stop)");
	result = kernel_echo_packets(stream, &rxq, &txq,
				     gpu_exit_condition,
				     gpu_latency_buf,
				     gpu_latency_count,
				     sample_cfg->cpu_proxy);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("kernel_echo_packets: %s", doca_error_get_descr(result));
		goto shutdown;
	}

	/* ---- Wait for Ctrl-C ---- */
	DOCA_LOG_INFO("Echo kernel running. Send UDP packets to NIC port %s",
		      sample_cfg->nic_pcie_addr);
	while (DOCA_GPUNETIO_VOLATILE(force_quit) == false)
		;

shutdown:
	printf("[SHUTDOWN] Ctrl+C caught, shutting down\n");
	fflush(stdout);

	/*
	 * Signal the GPU kernel to exit while the proxy threads are still alive.
	 * Without the barrier, the CPU write may sit in a WC buffer.
	 */
	DOCA_GPUNETIO_VOLATILE(*cpu_exit_condition) = 1;
	__sync_synchronize();

	/* Poll for kernel completion (2-second timeout) */
	printf("[SHUTDOWN] Waiting for GPU kernel to exit...\n");
	fflush(stdout);
	bool kernel_done = false;
	for (int i = 0; i < 2000; i++) {
		if (cudaStreamQuery(stream) == cudaSuccess) {
			kernel_done = true;
			break;
		}
		usleep(1000);
	}
	printf("[SHUTDOWN] GPU kernel %s\n",
	       kernel_done ? "exited cleanly" : "timed out, proceeding");
	fflush(stdout);

	/* Stop proxy threads */
	if (sample_cfg->cpu_proxy) {
		printf("[SHUTDOWN] Stopping proxy threads...\n");
		fflush(stdout);
		if (rxq_proxy_started) {
			WRITE_ONCE_64b(*rxq_proxy_args.exit_flag, 1);
			pthread_join(rxq_proxy_tid, NULL);
		}
		if (txq_proxy_started) {
			WRITE_ONCE_64b(*txq_proxy_args.exit_flag, 1);
			pthread_join(txq_proxy_tid, NULL);
		}
		free(rxq_proxy_args.exit_flag);
		free(txq_proxy_args.exit_flag);
		printf("[SHUTDOWN] Proxy threads stopped\n");
		fflush(stdout);
	}

	cudaStreamSynchronize(stream);

	/* ---- Print latency statistics ---- */
	print_latency_stats(cpu_latency_buf, cpu_latency_count[0]);

exit:
	destroy_txq(&txq);
	destroy_rxq(&rxq);

	if (gpu_dev != NULL) {
		result = doca_gpu_destroy(gpu_dev);
		if (result != DOCA_SUCCESS)
			DOCA_LOG_ERR("Failed to destroy GPU device: %s", doca_error_get_descr(result));
	}

	DOCA_LOG_INFO("Sample finished");
	return DOCA_SUCCESS;
}
