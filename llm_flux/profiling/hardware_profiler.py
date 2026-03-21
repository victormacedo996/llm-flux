"""
profiling/hardware_profiler.py — Hardware profiling domain service.

Retrieves CPU, GPU (via torch.cuda), and RAM information.
Currently supports Linux only; raise PlatformNotSupportedException otherwise.
"""
from __future__ import annotations

import platform
from datetime import datetime

import psutil
import torch
import torch.cuda

from llm_flux.profiling.types.hardware import (
    CPUInfo,
    GPUInfo,
    GPUMemoryInfo,
    GPUProperties,
    HardwareProfile,
    RAMInfo,
    SystemGPUInfo,
)


class PlatformNotSupportedException(Exception):
    """Raised when the host OS is not yet supported by HardwareProfiler."""


class HardwareProfiler:
    """
    Retrieves a snapshot of the host machine's hardware configuration.

    Usage::

        profile = HardwareProfiler().retrieve_hardware_information()
        print(profile.gpu.cuda_version)
    """

    # ── Public interface ──────────────────────────────────────────────────────

    def retrieve_hardware_information(self) -> HardwareProfile:
        return HardwareProfile(
            cpu=self.retrieve_cpu_information(),
            gpu=self.retrieve_gpu_information(),
            ram=self.retrieve_ram_information(),
        )

    def retrieve_cpu_information(self) -> CPUInfo:
        current_platform = platform.system()
        if current_platform == "Linux":
            return self._get_linux_cpu_info()
        raise PlatformNotSupportedException(
            f"Platform '{current_platform}' is not supported yet. "
            "Contributions welcome — implement a _get_<os>_cpu_info() method."
        )

    def retrieve_gpu_information(self) -> SystemGPUInfo:
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            return SystemGPUInfo(cuda_available=False, timestamp=datetime.now())

        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device() if device_count > 0 else None
        cuda_version = torch.version.cuda
        cudnn_version = (
            str(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None
        )

        gpus: list[GPUInfo] = []
        for device_id in range(device_count):
            try:
                gpus.append(self._get_gpu_info(device_id))
            except Exception as exc:  # pragma: no cover
                print(f"Warning: could not retrieve info for GPU {device_id}: {exc}")

        return SystemGPUInfo(
            cuda_available=True,
            cuda_version=cuda_version,
            cudnn_version=cudnn_version,
            device_count=device_count,
            current_device=current_device,
            gpus=gpus,
            timestamp=datetime.now(),
        )

    def retrieve_ram_information(self) -> RAMInfo:
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return RAMInfo(
            total_memory_gb=self._bytes_to_gb(vm.total),
            available_memory_gb=self._bytes_to_gb(vm.available),
            free_memory_gb=self._bytes_to_gb(vm.free),
            swap_total_gb=self._bytes_to_gb(swap.total),
            swap_free_gb=self._bytes_to_gb(swap.free),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_linux_cpu_info(self) -> CPUInfo:
        cpu_name = "Unknown"
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu_name = line.strip().split(": ", 1)[1]
                        break
        except (FileNotFoundError, IndexError):
            pass  # fallback to "Unknown"

        freq = psutil.cpu_freq()
        return CPUInfo(
            architecture=platform.processor(),
            name=cpu_name,
            max_freq=freq.max if freq else 0.0,
            platform=platform.platform(),
            physical_cores=psutil.cpu_count(logical=False) or 1,
            total_cores=psutil.cpu_count(logical=True) or 1,
        )

    def _get_gpu_info(self, device_id: int) -> GPUInfo:
        return GPUInfo(
            device_id=device_id,
            device_name=torch.cuda.get_device_name(device_id),
            is_available=True,
            memory_info=self._get_gpu_memory_info(device_id),
            properties=self._get_gpu_properties(device_id),
        )

    def _get_gpu_memory_info(self, device_id: int) -> GPUMemoryInfo:
        with torch.cuda.device(device_id):
            props = torch.cuda.get_device_properties(device_id)
            total = props.total_memory
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            free = total - allocated
            return GPUMemoryInfo(
                total=total, allocated=allocated,
                cached=reserved, reserved=reserved, free=free,
                total_gb=self._bytes_to_gb(total),
                allocated_gb=self._bytes_to_gb(allocated),
                cached_gb=self._bytes_to_gb(reserved),
                reserved_gb=self._bytes_to_gb(reserved),
                free_gb=self._bytes_to_gb(free),
            )

    def _get_gpu_properties(self, device_id: int) -> GPUProperties:
        p = torch.cuda.get_device_properties(device_id)
        return GPUProperties(
            name=p.name, major=p.major, minor=p.minor,
            total_memory=p.total_memory,
            multi_processor_count=p.multi_processor_count,
            max_threads_per_multi_processor=p.max_threads_per_multi_processor,
            max_threads_per_block=p.max_threads_per_block,
            max_block_dim=list(p.max_block_dim),
            max_grid_dim=list(p.max_grid_dim),
            warp_size=p.warp_size,
        )

    @staticmethod
    def _bytes_to_gb(value: int) -> float:
        return round(value / (1024 ** 3), 2)
