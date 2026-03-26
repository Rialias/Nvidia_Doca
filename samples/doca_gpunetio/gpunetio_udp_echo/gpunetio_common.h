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

#ifndef GPUNETIO_UDP_ECHO_COMMON_H_
#define GPUNETIO_UDP_ECHO_COMMON_H_

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <pthread.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <doca_error.h>
#include <doca_dev.h>
#include <doca_mmap.h>
#include <doca_gpunetio.h>
#include <doca_gpunetio_eth_def.h>
#include <doca_eth_rxq.h>
#include <doca_eth_rxq_gpu_data_path.h>
#include <doca_eth_txq.h>
#include <doca_eth_txq_gpu_data_path.h>
#include <doca_log.h>

#include "common.h"

/* ---- sizing ---- */
#define MAX_PCI_ADDRESS_LEN   32U
#define CUDA_BLOCK_THREADS    32
#define ETHER_ADDR_LEN        6

/* RXQ ring: how many packets can be in flight at once */
#define MAX_PKT_NUM           16384
#define MAX_PKT_SIZE          2048

/* Per rxq_recv call: max batch size and timeout before returning 0 pkts */
#define MAX_RX_NUM_PKTS       2048
#define MAX_RX_TIMEOUT_NS     50000   /* 50 µs */

/* TXQ descriptor ring depth */
#define MAX_SQ_DESCR_NUM      8192

/* How many latency samples to keep in GPU memory (ring) */
#define LATENCY_RING_SIZE     65536

/* ---- helpers ---- */
#define ALIGN_SIZE(size, align) size = ((size + (align)-1) / (align)) * (align)

#ifndef ACCESS_ONCE_64b
#define ACCESS_ONCE_64b(x)    (*(volatile uint64_t *)&(x))
#endif
#ifndef WRITE_ONCE_64b
#define WRITE_ONCE_64b(x, v)  (ACCESS_ONCE_64b(x) = (v))
#endif

/* ---- packet header structs ---- */
struct ether_hdr {
	uint8_t  d_addr_bytes[ETHER_ADDR_LEN];
	uint8_t  s_addr_bytes[ETHER_ADDR_LEN];
	uint16_t ether_type;
} __attribute__((__packed__));

struct ipv4_hdr {
	uint8_t  version_ihl;
	uint8_t  tos;
	uint16_t total_length;
	uint16_t packet_id;
	uint16_t fragment_offset;
	uint8_t  time_to_live;
	uint8_t  next_proto_id;
	uint16_t hdr_checksum;
	uint32_t src_addr;
	uint32_t dst_addr;
} __attribute__((__packed__));

struct udp_hdr {
	uint16_t src_port;
	uint16_t dst_port;
	uint16_t dgram_len;
	uint16_t dgram_cksum;
} __attribute__((__packed__));

/* Offsets from start of packet buffer */
#define ETH_HDR_SIZE   ((uint32_t)sizeof(struct ether_hdr))   /* 14 */
#define IPV4_HDR_SIZE  ((uint32_t)sizeof(struct ipv4_hdr))    /* 20 */
#define UDP_HDR_SIZE   ((uint32_t)sizeof(struct udp_hdr))      /* 8  */

/* ---- CPU proxy thread args ---- */
struct rxq_cpu_proxy_args {
	struct doca_eth_rxq *rxq;
	uint64_t *exit_flag;
};

struct txq_cpu_proxy_args {
	struct doca_eth_txq *txq;
	uint64_t *exit_flag;
};

/* ---- queue object structs ---- */
struct echo_rxq {
	struct doca_gpu         *gpu_dev;
	struct doca_dev         *ddev;
	struct doca_flow_port   *port;

	struct doca_ctx         *eth_rxq_ctx;
	struct doca_eth_rxq     *eth_rxq_cpu;
	struct doca_gpu_eth_rxq *eth_rxq_gpu;

	struct doca_mmap        *pkt_buff_mmap;
	void                    *gpu_pkt_addr;
	int                      dmabuf_fd;
	uint32_t                 rx_pkt_mkey;  /* NIC mkey for RX cyclic buf (network byte order) — used for zero-copy TX */

	struct doca_flow_pipe       *rxq_pipe;
	struct doca_flow_pipe       *root_pipe;
	struct doca_flow_pipe_entry *root_udp_entry;
};

struct echo_txq {
	struct doca_gpu         *gpu_dev;
	struct doca_dev         *ddev;

	struct doca_ctx         *eth_txq_ctx;
	struct doca_eth_txq     *eth_txq_cpu;
	struct doca_gpu_eth_txq *eth_txq_gpu;
};

/* ---- per-packet latency record written by GPU kernel ---- */
struct latency_sample {
	uint64_t rx_ts_ns;    /* GPU global timer just after rxq_recv returns (batch) */
	uint64_t tx_ts_ns;    /* GPU global timer just before txq_send */
	uint64_t latency_ns;  /* tx_ts_ns - rx_ts_ns = GPU processing time per batch */
};

/* ---- app config ---- */
struct sample_echo_cfg {
	char     gpu_pcie_addr[MAX_PCI_ADDRESS_LEN];
	char     nic_pcie_addr[MAX_PCI_ADDRESS_LEN];
	int      cuda_id;
	bool     cpu_proxy;
};

/* ---- function declarations ---- */
doca_error_t gpunetio_udp_echo(struct sample_echo_cfg *sample_cfg);

#if __cplusplus
extern "C" {
#endif

/*
 * Launch a CUDA kernel that receives UDP packets, echoes them back, and records latency.
 *
 * @stream [in]: CUDA stream
 * @rxq [in]: receive queue
 * @txq [in]: transmit queue
 * @gpu_exit_condition [in]: GPU-side exit flag (GPU_CPU mem)
 * @latency_buf [in]: GPU-side ring of latency_sample structs (CPU_GPU mem)
 * @latency_count [in]: running count of samples written (CPU_GPU mem)
 * @cpu_proxy [in]: true = use CPU_PROXY NIC handler for both RXQ and TXQ
 * @return: DOCA_SUCCESS on success
 */
doca_error_t kernel_echo_packets(cudaStream_t stream,
				 struct echo_rxq *rxq,
				 struct echo_txq *txq,
				 uint32_t *gpu_exit_condition,
				 struct latency_sample *latency_buf,
				 uint64_t *latency_count,
				 bool cpu_proxy);

#if __cplusplus
}
#endif
#endif /* GPUNETIO_UDP_ECHO_COMMON_H_ */
