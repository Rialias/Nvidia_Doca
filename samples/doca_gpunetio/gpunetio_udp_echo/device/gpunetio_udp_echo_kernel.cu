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

#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>
#include <doca_log.h>
#include "gpunetio_common.h"

DOCA_LOG_REGISTER(GPUNETIO_UDP_ECHO::KERNEL);

/*
 * Rewrite Ethernet/IP/UDP headers in-place so the packet becomes an echo reply:
 * swap src↔dst MAC, swap src↔dst IP, swap src↔dst UDP port, reset TTL, zero checksums.
 * Called by thread 0 only.
 */
__device__ static void rewrite_echo_headers(uint8_t *pkt)
{
	struct ether_hdr *eth = (struct ether_hdr *)pkt;
	struct ipv4_hdr  *ip  = (struct ipv4_hdr  *)(pkt + ETH_HDR_SIZE);
	struct udp_hdr   *udp = (struct udp_hdr   *)(pkt + ETH_HDR_SIZE + IPV4_HDR_SIZE);
	uint8_t  tmp8;
	uint16_t tmp16;
	uint32_t tmp32;

	/* Swap MAC addresses */
	for (int i = 0; i < ETHER_ADDR_LEN; i++) {
		tmp8 = eth->d_addr_bytes[i];
		eth->d_addr_bytes[i] = eth->s_addr_bytes[i];
		eth->s_addr_bytes[i] = tmp8;
	}

	/* Swap IP addresses, reset TTL, zero IP checksum (NIC offload or ignored in loopback) */
	tmp32 = ip->src_addr;
	ip->src_addr      = ip->dst_addr;
	ip->dst_addr      = tmp32;
	ip->time_to_live  = 64;
	ip->hdr_checksum  = 0;

	/* Swap UDP ports, zero UDP checksum */
	tmp16 = udp->src_port;
	udp->src_port    = udp->dst_port;
	udp->dst_port    = tmp16;
	udp->dgram_cksum = 0;
}

/*
 * GPU echo kernel.
 *
 * One CUDA block of CUDA_BLOCK_THREADS (32) threads.
 *
 * --- Loop body ---
 * 1. RECEIVE (block scope): all 32 threads cooperate to receive up to
 *    CUDA_BLOCK_THREADS packets in one rxq_recv call (500 µs timeout).
 *
 * 2. PROCESS: for each received packet k (handled by thread k):
 *      a. Thread k rewrites Ethernet/IP/UDP headers in-place in the RX cyclic
 *         buffer (zero-copy — no separate TX buffer).
 *      b. Thread k records GPU timer → tx_ts.
 *      c. Thread k posts a TXQ WQE pointing directly at rx_addr, using the
 *         RX buffer's NIC mkey.  Last sender sets SEND_FLAG_NOTIFY for one CQE.
 *      d. Thread k records latency = tx_ts - rx_batch_ts.
 *
 * 3. POLL: thread 0 waits for that CQE — ensuring the NIC has finished reading
 *    the RX buffer before rxq_recv can advance the ring and reuse those slots.
 *
 * --- Latency ---
 * Both timestamps are GPU global timer readings (same clock, no synchronization
 * required).  The NIC hardware RX timestamp is intentionally NOT used here:
 * it runs on a different, unsynchronized clock and would produce meaningless results.
 *
 * rx_ts_ns = GPU global timer sampled by thread 0 immediately after rxq_recv
 *            returns (stored in shared memory so all threads see it).
 * tx_ts_ns = GPU global timer sampled by each thread just before its txq_send.
 * latency  = tx_ts_ns - rx_ts_ns  ≈  rxq_recv-wakeup → memcpy → header-rewrite
 *            → WQE post  (pure GPU processing time per batch).
 *
 * template param nic_handler:
 *   CPU_PROXY → GPU does not ring NIC doorbell; CPU proxy threads handle UAR
 *               ring for both RXQ (doca_eth_rxq_gpu_cpu_proxy_progress) and
 *               TXQ (doca_eth_txq_gpu_cpu_proxy_progress).
 *   AUTO      → GPU SM rings doorbell directly (requires PIX/PHB PCIe topology).
 */
template <enum doca_gpu_dev_eth_nic_handler nic_handler = DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO>
__global__ void echo_kernel(
	struct doca_gpu_eth_rxq *rxq,
	struct doca_gpu_eth_txq *txq,
	const uint32_t           rx_mkey,      /* NIC mkey for RX cyclic buffer (network byte order) */
	uint32_t                *exit_cond,
	struct latency_sample   *latency_buf,
	uint64_t                *latency_count)
{
	__shared__ uint64_t out_first_pkt_idx;
	__shared__ uint32_t out_pkt_num;
	__shared__ struct doca_gpu_dev_eth_rxq_attr out_attr[CUDA_BLOCK_THREADS];
	__shared__ uint64_t rx_batch_ts;  /* GPU timer sampled right after rxq_recv returns */

	doca_error_t ret;
	uint32_t num_completed;

	while (DOCA_GPUNETIO_VOLATILE(*exit_cond) == 0) {

		/* -------- RECEIVE: all threads cooperate -------- */
		ret = doca_gpu_dev_eth_rxq_recv<DOCA_GPUNETIO_ETH_EXEC_SCOPE_BLOCK,
						DOCA_GPUNETIO_ETH_MCST_AUTO,
						nic_handler,
						DOCA_GPUNETIO_ETH_RX_ATTR_ALL>(
				rxq,
				CUDA_BLOCK_THREADS,   /* max one packet per thread */
				MAX_RX_TIMEOUT_NS,
				&out_first_pkt_idx,
				&out_pkt_num,
				out_attr);

		if (ret != DOCA_SUCCESS) {
			if (threadIdx.x == 0) {
				printf("[ECHO KERNEL] rxq_recv error %d\n", (int)ret);
				DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
			}
			break;
		}

		if (out_pkt_num == 0)
			continue;

		/*
		 * Sample GPU timer immediately after rxq_recv — this is as close as we
		 * can get to "when the GPU kernel woke up with packets".
		 * Thread 0 writes to shared memory; __syncthreads below distributes it.
		 */
		if (threadIdx.x == 0)
			rx_batch_ts = doca_gpu_dev_eth_query_globaltimer();
		__syncthreads();

		/* -------- PROCESS: each thread handles one packet -------- */
		if ((uint32_t)threadIdx.x < out_pkt_num) {
			const uint32_t k         = threadIdx.x;
			uint64_t       rx_addr   = doca_gpu_dev_eth_rxq_get_pkt_addr(rxq, out_first_pkt_idx + k);
			const uint32_t pkt_bytes = out_attr[k].bytes;
			const uint64_t rx_ts     = rx_batch_ts;  /* GPU-side RX time (same clock as tx_ts) */

			/* --- Rewrite headers in-place in the RX cyclic buffer (zero-copy) --- */
			rewrite_echo_headers((uint8_t *)rx_addr);

			/* --- Record GPU timestamp before posting WQE (same clock as rx_batch_ts) --- */
			const uint64_t tx_ts = doca_gpu_dev_eth_query_globaltimer();

			/*
			 * --- Send ---
			 * RESOURCE_SHARING_MODE_GPU: atomic WQE slot allocation — safe for
			 * multiple threads calling independently with EXEC_SCOPE_THREAD.
			 * Only the last sending thread (k == pkt_num-1) sets NOTIFY so the
			 * NIC generates exactly one CQE for this entire batch.
			 */
			const bool is_last_sender = (k == (out_pkt_num - 1u));
			const enum doca_gpu_eth_send_flags flags = is_last_sender
				? DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY
				: DOCA_GPUNETIO_ETH_SEND_FLAG_NONE;

			doca_gpu_dev_eth_ticket_t ticket;
			doca_gpu_dev_eth_txq_send<DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
						  DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU,
						  nic_handler,
						  DOCA_GPUNETIO_ETH_EXEC_SCOPE_THREAD>(
				txq,
				rx_addr,
				rx_mkey,
				(size_t)pkt_bytes,
				flags,
				&ticket);

			/* --- Record latency sample in ring buffer --- */
			const uint64_t lat  = (tx_ts >= rx_ts) ? (tx_ts - rx_ts) : 0ULL;
			const uint64_t slot = atomicAdd((unsigned long long *)latency_count, 1ULL)
					      % LATENCY_RING_SIZE;
			latency_buf[slot].rx_ts_ns   = rx_ts;
			latency_buf[slot].tx_ts_ns   = tx_ts;
			latency_buf[slot].latency_ns = lat;
		}

		__syncthreads();

		/*
		 * Thread 0 polls for the single CQE generated by the last sender's NOTIFY.
		 * Blocking wait (WAIT_FLAG_B): thread 0 spins in GPU until NIC writes the CQE.
		 * This ensures TX completions are reaped before the next recv call overwrites
		 * the tx_pkt_buf slots (max rate: recv batch → echo batch → recv batch …).
		 */
		if (threadIdx.x == 0) {
			ret = doca_gpu_dev_eth_txq_poll_completion<DOCA_GPUNETIO_ETH_CQ_POLL_LAST>(
				txq,
				1,  /* 1 NOTIFY per batch */
				DOCA_GPUNETIO_ETH_WAIT_FLAG_B,
				&num_completed);
			if (ret != DOCA_SUCCESS)
				DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
		}

		__syncthreads();
	}

	/* Flush latency_count write to system memory so CPU can read it */
	if (threadIdx.x == 0)
		__threadfence_system();
}

extern "C" {

doca_error_t kernel_echo_packets(cudaStream_t stream,
				 struct echo_rxq *rxq,
				 struct echo_txq *txq,
				 uint32_t *gpu_exit_condition,
				 struct latency_sample *latency_buf,
				 uint64_t *latency_count,
				 bool cpu_proxy)
{
	cudaError_t result;

	if (rxq == NULL || txq == NULL || gpu_exit_condition == NULL
	    || latency_buf == NULL || latency_count == NULL) {
		DOCA_LOG_ERR("kernel_echo_packets: invalid input values");
		return DOCA_ERROR_INVALID_VALUE;
	}

	result = cudaGetLastError();
	if (result != cudaSuccess) {
		DOCA_LOG_ERR("[%s:%d] CUDA error before launch: %s",
			     __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	if (cpu_proxy) {
		echo_kernel<DOCA_GPUNETIO_ETH_NIC_HANDLER_CPU_PROXY>
			<<<1, CUDA_BLOCK_THREADS, 0, stream>>>(
				rxq->eth_rxq_gpu,
				txq->eth_txq_gpu,
				rxq->rx_pkt_mkey,
				gpu_exit_condition,
				latency_buf,
				latency_count);
	} else {
		echo_kernel<DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO>
			<<<1, CUDA_BLOCK_THREADS, 0, stream>>>(
				rxq->eth_rxq_gpu,
				txq->eth_txq_gpu,
				rxq->rx_pkt_mkey,
				gpu_exit_condition,
				latency_buf,
				latency_count);
	}

	result = cudaGetLastError();
	if (result != cudaSuccess) {
		DOCA_LOG_ERR("[%s:%d] CUDA error after launch: %s",
			     __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	return DOCA_SUCCESS;
}

} /* extern "C" */
