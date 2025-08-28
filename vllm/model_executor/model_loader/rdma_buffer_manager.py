# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDMA Buffer Manager for handling pre-registered memory buffers."""

import logging
from typing import List, Tuple, Optional
import numpy as np
import multiprocessing.shared_memory as shm

logger = logging.getLogger(__name__)


class RDMABufferManager:
    """管理RDMA预注册内存缓冲区的类.
    
    这个类负责创建、管理和清理RDMA预注册内存缓冲区，
    支持同步和异步RDMA操作的内存需求。
    """
    
    def __init__(self):
        """初始化RDMA缓冲区管理器."""
        self._buffers: List[np.ndarray] = []
        self._buffer_ptrs: List[int] = []
        self._shared_memory_objects: List[shm.SharedMemory] = []
        self._shared_memory_names: List[str] = []
        self._max_file_size: int = 0
        self._pipeline_window_size: int = 0
        self._is_initialized: bool = False
        self._use_shared_memory: bool = False
        self._rdma_client = None
    
    @property
    def is_initialized(self) -> bool:
        """检查缓冲区是否已初始化."""
        return self._is_initialized
    
    @property
    def buffers(self) -> List[np.ndarray]:
        """获取预注册缓冲区列表."""
        return self._buffers
    
    @property
    def buffer_ptrs(self) -> List[int]:
        """获取预注册缓冲区指针列表."""
        return self._buffer_ptrs
    
    @property
    def max_file_size(self) -> int:
        """获取最大文件大小."""
        return self._max_file_size
    
    @property
    def pipeline_window_size(self) -> int:
        """获取流水线窗口大小."""
        return self._pipeline_window_size
    
    @property
    def use_shared_memory(self) -> bool:
        """检查是否使用共享内存."""
        return self._use_shared_memory
    
    @property
    def shared_memory_names(self) -> List[str]:
        """获取共享内存名称列表."""
        return self._shared_memory_names
    
    def initialize(self, rdma_client: object, max_file_size: int, pipeline_window_size: int, use_shared_memory: bool = False) -> None:
        """初始化持久化预注册缓冲区.
        
        Args:
            rdma_client: RDMA客户端实例
            max_file_size: 最大文件大小（字节）
            pipeline_window_size: 流水线窗口大小
            use_shared_memory: 是否使用共享内存
            
        Raises:
            RuntimeError: 当缓冲区初始化失败时抛出
        """
        if self._is_initialized:
            logger.info("Buffer manager already initialized, skipping initialization")
            return
        
        self._rdma_client = rdma_client
        self._max_file_size = max_file_size
        self._pipeline_window_size = pipeline_window_size
        self._use_shared_memory = use_shared_memory
        
        try:
            # 为每个流水线槽位分配单独的缓冲区
            self._buffers = []
            self._buffer_ptrs = []
            self._shared_memory_objects = []
            self._shared_memory_names = []
            
            if use_shared_memory:
                # 创建共享内存缓冲区
                import uuid
                for i in range(self._pipeline_window_size):
                    # 生成唯一的共享内存名称
                    shm_name = f"vllm_rdma_shm_{uuid.uuid4().hex}_{i}"
                    # 创建共享内存对象
                    shm_obj = shm.SharedMemory(create=True, size=self._max_file_size, name=shm_name)
                    # 将共享内存包装为numpy数组以便操作 []
                    shared_buffer = np.ndarray(shape=(self._max_file_size,), dtype=np.uint8, buffer=shm_obj.buf)
                    # shared_buffer = np.frombuffer(shm.buf, dtype=np.uint8) 
                    # 保存共享内存对象引用，防止被垃圾回收
                    self._shared_memory_objects.append(shm_obj)
                    self._shared_memory_names.append(shm_name)
                    
                    # 获取共享内存缓冲区的地址
                    # import ctypes
                    # buffer_ptr = ctypes.addressof(ctypes.c_char.from_buffer(shm_obj.buf))
                    buffer_ptr = shared_buffer.ctypes.data
                    self._buffer_ptrs.append(buffer_ptr)
                
                logger.info(f"Created {self._pipeline_window_size} shared memory buffers of size {self._max_file_size} bytes each")
            else:
                # 先分配所有缓冲区
                for i in range(self._pipeline_window_size):
                    # 预分配缓冲区
                    buffer = np.empty(self._max_file_size, dtype=np.uint8)
                    buffer_ptr = buffer.ctypes.data
                    
                    self._buffers.append(buffer)
                    self._buffer_ptrs.append(buffer_ptr)
            
            # 批量注册所有内存
            capacities = [self._max_file_size] * len(self._buffer_ptrs)
            ret = self._rdma_client.batch_register_memory(self._buffer_ptrs, capacities)
            if ret != 0:
                raise RuntimeError(
                    f"Failed to batch register memory for {self._pipeline_window_size} buffers, "
                    f"return code: {ret}"
                )
                
        except Exception as e:
            # 清理已分配的资源
            self.cleanup()
            raise RuntimeError(
                f"Failed to allocate and register persistent buffer of size "
                f"{self._max_file_size} for {self._pipeline_window_size} slots: {e}"
            )
        
        self._is_initialized = True
        logger.info(
            f"Initialized {self._pipeline_window_size} persistent buffers of size "
            f"{self._max_file_size} bytes each "
            f"({self._max_file_size / (1024*1024*1024):.2f} GB each)"
        )
    
    def get_buffer(self, index: int) -> Tuple[np.ndarray, int]:
        """获取指定索引的缓冲区和指针.
        
        Args:
            index: 缓冲区索引
            
        Returns:
            Tuple of (buffer, buffer_pointer)
            
        Raises:
            IndexError: 当索引超出范围时抛出
            RuntimeError: 当缓冲区未初始化时抛出
        """
        if not self._is_initialized:
            raise RuntimeError("Buffer manager not initialized")
        
        if index < 0 or index >= len(self._buffers):
            raise IndexError(f"Buffer index {index} out of range [0, {len(self._buffers)})")
        
        return self._buffers[index], self._buffer_ptrs[index]
    
    def get_buffer_by_slot(self, slot: int) -> Tuple[np.ndarray, int]:
        """根据流水线槽位获取缓冲区和指针.
        
        Args:
            slot: 流水线槽位号
            
        Returns:
            Tuple of (buffer, buffer_pointer)
        """
        index = slot % self._pipeline_window_size
        return self.get_buffer(index)
    
    def cleanup(self) -> None:
        """清理预注册缓冲区.
        
        注意：Transfer Engine 在退出时不会释放我们在 Python 中申请的内存。
        我们需要在 Python 代码中自己管理内存的生命周期。
        
        此方法会先调用 unregister_memory 接口注销内存注册，然后再清理缓冲区引用。
        实际的内存释放由 Python 的垃圾回收机制处理。
        """
        if not self._is_initialized:
            return
        
        # 批量注销所有已注册的内存
        if self._rdma_client is not None and self._buffer_ptrs:
            try:
                # 调用 batch_unregister_memory 接口批量注销内存注册
                ret = self._rdma_client.batch_unregister_memory(self._buffer_ptrs)
                if ret != 0:
                    logger.warning(f"Failed to batch unregister memory, return code: {ret}")
            except Exception as e:
                logger.warning(f"Failed to batch unregister memory: {e}")
        
        # 清理缓冲区引用
        self._buffers.clear()
        self._buffer_ptrs.clear()
        
        # 清理共享内存对象
        if self._use_shared_memory:
            for shm_obj in self._shared_memory_objects:
                try:
                    shm_obj.close()
                    shm_obj.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up shared memory object: {e}")
            self._shared_memory_objects.clear()
            self._shared_memory_names.clear()
        
        # 重置状态
        self._max_file_size = 0
        self._pipeline_window_size = 0
        self._is_initialized = False
        self._use_shared_memory = False
        self._rdma_client = None
        
        logger.info("RDMA buffer manager cleaned up")
    
    def __enter__(self):
        """上下文管理器入口."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，确保资源清理."""
        self.cleanup()


# 向后兼容的全局缓冲区管理器实例
# 在重构代码时应逐步移除对此全局实例的依赖
_global_buffer_manager: Optional[RDMABufferManager] = None


def get_global_buffer_manager() -> RDMABufferManager:
    """获取全局缓冲区管理器实例（用于向后兼容）.
    
    Returns:
        RDMABufferManager实例
    """
    global _global_buffer_manager
    if _global_buffer_manager is None:
        _global_buffer_manager = RDMABufferManager()
    return _global_buffer_manager


def cleanup_global_buffer_manager() -> None:
    """清理全局缓冲区管理器实例."""
    global _global_buffer_manager
    if _global_buffer_manager is not None:
        _global_buffer_manager.cleanup()
        _global_buffer_manager = None
