"""Pydantic types for hardware profiling results."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class CPUInfo(BaseModel):
    name: str
    architecture: str
    platform: str
    physical_cores: int
    total_cores: int
    max_freq: float  # MHz


class GPUMemoryInfo(BaseModel):
    total: int
    allocated: int
    cached: int
    reserved: int
    free: int
    total_gb: float
    allocated_gb: float
    cached_gb: float
    reserved_gb: float
    free_gb: float


class GPUProperties(BaseModel):
    name: str
    major: int
    minor: int
    total_memory: int
    multi_processor_count: int
    max_threads_per_multi_processor: int
    max_threads_per_block: int
    max_block_dim: List[int]
    max_grid_dim: List[int]
    warp_size: int


class GPUInfo(BaseModel):
    device_id: int
    device_name: str
    is_available: bool
    memory_info: GPUMemoryInfo
    properties: GPUProperties


class SystemGPUInfo(BaseModel):
    cuda_available: bool
    cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None
    device_count: int = 0
    current_device: Optional[int] = None
    gpus: List[GPUInfo] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class RAMInfo(BaseModel):
    total_memory_gb: float
    available_memory_gb: float
    free_memory_gb: float
    swap_total_gb: float
    swap_free_gb: float


class HardwareProfile(BaseModel):
    cpu: CPUInfo
    gpu: SystemGPUInfo
    ram: RAMInfo
