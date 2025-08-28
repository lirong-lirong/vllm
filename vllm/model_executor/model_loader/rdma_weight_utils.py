# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for loading model weights via RDMA directly from remote storage."""
import json
import os
from collections.abc import Generator
from typing import Dict, Any, Tuple, List, Optional

import torch
from safetensors.torch import load, safe_open

from vllm.logger import init_logger
from vllm.model_executor.model_loader.rdma_buffer_manager import RDMABufferManager

logger = init_logger(__name__)

try:
    import etcd3
    ETCD_AVAILABLE = True
except ImportError:
    ETCD_AVAILABLE = False
    logger.warning("etcd3 library not available. RDMA weight loading may not work correctly.")


def rdma_safetensors_weights_iterator(
    model_name: str,
    rdma_client: Any,  # Mooncake Transfer Engine client
    etcd_host: str = "localhost",
    etcd_port: int = 2379,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """使用safetensors.load()从RDMA读取的数据加载权重
    
    Args:
        model_name: 模型名称
        rdma_client: Mooncake Transfer Engine client for RDMA operations
        etcd_host: ETCD服务器主机名
        etcd_port: ETCD服务器端口
        
    Yields:
        Tuples of (weight_name, weight_tensor)
    """
    if not ETCD_AVAILABLE:
        raise RuntimeError("ETCD library not available. Cannot discover model weights.")
    
    # 从环境变量获取配置
    use_async = os.getenv("VLLM_RDMA_ASYNC_LOADING", "false").lower() in ("true", "1", "yes")
    use_batch = os.getenv("VLLM_RDMA_BATCH_LOADING", "false").lower() in ("true", "1", "yes")
    use_safe_open = os.getenv("VLLM_RDMA_SAFE_OPEN_LOADING", "false").lower() in ("true", "1", "yes")
    pipeline_window_size = int(os.getenv("VLLM_RDMA_PIPELINE_WINDOW_SIZE", "3"))
    
    logger.info(f"Starting RDMA weight loading for model: {model_name}")
    logger.info(f"Async loading: {use_async}, Batch loading: {use_batch}, SafeOpen loading: {use_safe_open}, Pipeline window size: {pipeline_window_size}")
    
    # 初始化RDMA加载
    init_result = _initialize_rdma_loading(etcd_host, etcd_port, model_name)
    etcd_client, weight_files, file_metadata_cache, file_sizes, file_server_info, server_name = init_result
    
    # 如果启用批量下载，使用批量下载实现
    if use_batch:
        yield from _load_weights_batch(
            rdma_client, weight_files, file_sizes, file_server_info, 
            server_name
        )
    elif use_safe_open:
        yield from _load_weights_async_safe_open(
            rdma_client, weight_files, file_sizes, file_server_info, 
            server_name, pipeline_window_size
        )
    elif use_async:
        yield from _load_weights_async(
            rdma_client, weight_files, file_sizes, file_server_info, 
            server_name, pipeline_window_size
        )
    else:
        yield from _load_weights_sync(
            rdma_client, weight_files, file_sizes, file_server_info, 
            server_name
        )
    
    logger.info(f"Completed RDMA weight loading for model: {model_name}")


def _initialize_rdma_loading(
    etcd_host: str, 
    etcd_port: int, 
    model_name: str
) -> Tuple[Any, List[str], Dict[str, Dict], Dict[str, int], Dict[str, Dict], str]:
    """初始化RDMA加载过程
    
    Returns:
        Tuple of (etcd_client, weight_files, file_metadata_cache, file_sizes, file_server_info, server_name)
    """
    # 1. 连接到ETCD
    etcd_client = _connect_to_etcd(etcd_host, etcd_port)
    if not etcd_client:
        raise RuntimeError(f"Failed to connect to ETCD at {etcd_host}:{etcd_port}")
    
    # 2. 获取模型的所有权重文件
    weight_files = _get_model_weight_files(etcd_client, model_name)
    if not weight_files:
        raise RuntimeError(f"No weight files found for model {model_name}")
    
    logger.info(f"Found {len(weight_files)} weight files for model {model_name}")
    
    # 3. 预先获取所有权重文件的元数据，避免在循环中重复访问ETCD
    file_metadata_cache = {}
    file_sizes = {}
    file_server_info = {}
    
    for weight_file in weight_files:
        metadata = _get_weight_file_metadata(etcd_client, weight_file)
        if metadata:
            file_metadata_cache[weight_file] = metadata
            # 提取文件大小
            file_size = (
                metadata.get("size") or 
                metadata.get("Size") or 
                metadata.get("total_size") or 
                metadata.get("TotalSize") or
                1024 * 1024 * 100  # 默认100MB
            )
            file_sizes[weight_file] = file_size
            
            # 提取服务器信息和偏移量
            server_info = _extract_server_info_from_metadata(metadata)
            if server_info:
                file_server_info[weight_file] = server_info
    
    # 根据文件大小降序排序权重文件，以优化文件系统操作（避免扩容）
    logger.info("Sorting weight files by size in descending order for performance optimization.")
    weight_files.sort(key=lambda f: file_sizes.get(f, 0), reverse=True)
    
    # 4. 从权重文件中提取服务器信息
    servers = _extract_servers_from_weight_files_with_cache(etcd_client, weight_files, file_metadata_cache)
    if not servers:
        raise RuntimeError("No servers found for weight files")
    
    logger.info(f"Discovered {len(servers)} servers: {servers}")
    
    # 5. 使用第一个服务器进行连接测试和权重加载
    server_name = servers[0]
    logger.info(f"Using server {server_name} for RDMA weight loading")
    
    return etcd_client, weight_files, file_metadata_cache, file_sizes, file_server_info, server_name


def _load_weights_sync(
    rdma_client: Any,
    weight_files: List[str],
    file_sizes: Dict[str, int],
    file_server_info: Dict[str, Dict],
    server_name: str
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """同步加载权重文件（强制使用预注册内存）
    
    Yields:
        Tuples of (weight_name, weight_tensor)
    """
    import time
    
    # 记录初始化缓冲区时间
    init_start_time = time.perf_counter()
    
    max_file_size = max(file_sizes.values()) if file_sizes else 1024 * 1024 * 100
    # 创建缓冲区管理器实例
    buffer_manager = RDMABufferManager()
    buffer_manager.initialize(rdma_client, max_file_size, 1)
    buffers = buffer_manager.buffers
    sync_buffer = buffers[0]
    sync_buffer_ptr = buffer_manager.buffer_ptrs[0]
    
    init_end_time = time.perf_counter()
    init_time = init_end_time - init_start_time
    _log_performance_metrics("Buffer initialization", 0, init_time)
    
    logger.info(f"Using shared buffer of size {max_file_size} bytes ({max_file_size / (1024*1024*1024):.2f} GB) for sync loading")
    
    total_bytes = 0
    total_transfer_time = 0.0
    total_load_time = 0.0
    
    for weight_file in weight_files:
        logger.info(f"Loading weights from {weight_file} via RDMA (sync with pre-registered buffer)")
        
        try:
            transfer_start_time = time.perf_counter()
            
            file_size = file_sizes.get(weight_file, 1024 * 1024 * 100)  # 默认100MB
            server_info = file_server_info.get(weight_file, {"segment_name": server_name})
            offset = server_info.get("offset", 0)
            
            file_data = _read_weight_range_optimized(rdma_client, server_info, weight_file, offset, file_size, sync_buffer, sync_buffer_ptr)
            
            transfer_end_time = time.perf_counter()
            transfer_time = transfer_end_time - transfer_start_time
            total_transfer_time += transfer_time
            total_bytes += len(file_data)
            
            _log_performance_metrics("File transfer", len(file_data), transfer_time, f"file: {weight_file}")
            
            # 记录safetensors.load()时间
            load_start_time = time.perf_counter()
            
            # 使用safetensors.load()从内存数据加载所有张量
            tensors = load(file_data)
            
            load_end_time = time.perf_counter()
            load_time = load_end_time - load_start_time
            total_load_time += load_time
            
            _log_performance_metrics("File load", len(file_data), load_time, f"file: {weight_file}")
            
            tensor_count = 0
            for name, tensor in tensors.items():
                if name != "__metadata__":
                    yield name, tensor
                    tensor_count += 1
            
            logger.info(f"Loaded {tensor_count} tensors from {weight_file}")
        except Exception as e:
            logger.error(f"Failed to load weights from {weight_file}: {e}")
            raise
        finally:
            # 清理缓冲区管理器
            buffer_manager.cleanup()
    
    # 计算总体带宽
    _log_performance_metrics("Overall sync loading transfer", total_bytes, total_transfer_time)
    _log_performance_metrics("Overall sync loading load", total_bytes, total_load_time)


# 异步读取任务状态
class AsyncReadTaskOptimized:
    def __init__(self, weight_file, batch_id, buffer_addr, buffer_index, length, server_info):
        self.weight_file = weight_file
        self.batch_id = batch_id
        self.buffer_addr = buffer_addr
        self.buffer_index = buffer_index
        self.length = length
        self.server_info = server_info


def _wait_for_async_read_completion_optimized(rdma_client, batch_id):
    """等待异步读取完成函数"""
    import time
    
    # 轮询等待传输完成
    while True:
        logger.info("wait")
        status = rdma_client.get_batch_transfer_status([batch_id])
        if status == 0:  # 传输成功完成
            break
        elif status < 0:  # 传输失败
            raise RuntimeError(f"Async read failed with status {status}")
        time.sleep(0.001)


def _calculate_bandwidth(bytes_transferred: int, time_seconds: float) -> float:
    """计算带宽 (GB/s)
    
    Args:
        bytes_transferred: 传输的字节数
        time_seconds: 传输所用的时间（秒）
        
    Returns:
        带宽 (GB/s)
    """
    if time_seconds <= 0:
        return 0.0
    # 转换为GB/s: bytes / (1024^3) / seconds
    return (bytes_transferred / (1024 * 1024 * 1024)) / time_seconds


def _log_performance_metrics(operation: str, bytes_transferred: int, time_seconds: float, additional_info: str = ""):
    """记录性能指标
    
    Args:
        operation: 操作名称
        bytes_transferred: 传输的字节数
        time_seconds: 花费的时间（秒）
        additional_info: 附加信息
    """
    if time_seconds > 0:
        bandwidth = _calculate_bandwidth(bytes_transferred, time_seconds)
        if additional_info:
            logger.info(f"{operation} bandwidth: {bandwidth:.2f} GB/s, time spent: {time_seconds:.2f} seconds, {additional_info}")
        else:
            logger.info(f"{operation} bandwidth: {bandwidth:.2f} GB/s, time spent: {time_seconds:.2f} seconds")
    else:
        if additional_info:
            logger.info(f"{operation} time spent: {time_seconds:.2f} seconds, {additional_info}")
        else:
            logger.info(f"{operation} time spent: {time_seconds:.2f} seconds")



def _load_weights_async(
    rdma_client: Any,
    weight_files: List[str],
    file_sizes: Dict[str, int],
    file_server_info: Dict[str, Dict],
    server_name: str,
    pipeline_window_size: int
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """异步加载权重文件
    Yields:
        Tuples of (weight_name, weight_tensor)
    """
    import time
    from collections import deque
    
    # 记录初始化缓冲区时间
    init_start_time = time.perf_counter()
    
    # 初始化持久化缓冲区
    max_file_size = max(file_sizes.values()) if file_sizes else 1024 * 1024 * 100
    # 创建缓冲区管理器实例
    buffer_manager = RDMABufferManager()
    try:
        buffer_manager.initialize(rdma_client, max_file_size, pipeline_window_size)
        buffers = buffer_manager.buffers
    except Exception as e:
        raise RuntimeError(f"Failed to initialize buffer manager for async loading: {e}")
    
    init_end_time = time.perf_counter()
    init_time = init_end_time - init_start_time
    _log_performance_metrics("Buffer initialization", 0, init_time)
    
    # 使用双端队列存储正在进行的任务
    active_tasks = deque()
    
    # 初始化流水线：提交初始的预取请求
    init_transfer_start_time = time.perf_counter()
    init_transfer_bytes = 0
    
    # 提交初始窗口大小的异步读取请求
    for i in range(min(pipeline_window_size, len(weight_files))):
        weight_file = weight_files[i]
        file_size = file_sizes.get(weight_file, 1024 * 1024 * 100)
        server_info = file_server_info.get(weight_file, {"segment_name": server_name})
        offset = server_info.get("offset", 0)
        
        # 直接从数组中获取缓冲区地址
        buffer_ptr = buffer_manager.buffer_ptrs[i]
        
        # 提交异步读取请求（使用优化版本）
        try:
            batch_id, returned_buffer_addr, length = _read_weight_range_async_optimized(
                rdma_client, server_info, weight_file, offset, file_size,
                buffer_ptr
            )
            
            active_tasks.append(AsyncReadTaskOptimized(
                weight_file=weight_file,
                batch_id=batch_id,
                buffer_addr=returned_buffer_addr,
                buffer_index=i,  # 存储缓冲区索引而不是地址
                length=length,
                server_info=server_info
            ))
            init_transfer_bytes += length
        except Exception as e:
            logger.error(f"Failed to submit async read request for {weight_file}: {e}")
            # 清理缓冲区管理器
            buffer_manager.cleanup()
            # 错误情况下不需要释放缓冲区，因为使用的是预注册的大块内存
            raise
    
    init_transfer_end_time = time.perf_counter()
    init_transfer_time = init_transfer_end_time - init_transfer_start_time
    _log_performance_metrics("Initial async transfer", init_transfer_bytes, init_transfer_time)
    
    # 处理所有权重文件
    next_submit_index = pipeline_window_size
    total_bytes = 0
    total_transfer_time = 0.0
    total_load_time = 0.0
    
    try:
        # 继续处理直到所有任务完成
        while active_tasks:
            # 获取最早提交的任务
            current_task = active_tasks.popleft()
            weight_file = current_task.weight_file
            batch_id = current_task.batch_id
            buffer_index = current_task.buffer_index  # 使用缓冲区索引
            length = current_task.length
            server_info = current_task.server_info
            
            logger.info(f"Processing weights from {weight_file} via RDMA (async optimized)")
            
            try:
                # 记录开始时间
                transfer_start_time = time.perf_counter()
                
                # 等待当前文件传输完成（使用优化版本）
                _wait_for_async_read_completion_optimized(
                    rdma_client, batch_id
                )
                
                buffer = buffer_manager.buffers[buffer_index]
                file_data = buffer[:length].tobytes()
                # file_data = memoryview(buffer)[:length].tobytes()
                # file_data = memoryview(buffer.data)[0:length]
                
                # 记录结束时间和计算带宽
                transfer_end_time = time.perf_counter()
                transfer_time = transfer_end_time - transfer_start_time
                total_transfer_time += transfer_time
                total_bytes += len(file_data)
                
                _log_performance_metrics("File transfer", len(file_data), transfer_time, f"file: {weight_file}")
                
                # 记录safetensors.load()时间
                load_start_time = time.perf_counter()
                
                tensors = load(file_data)
                
                load_end_time = time.perf_counter()
                load_time = load_end_time - load_start_time
                total_load_time += load_time
                
                _log_performance_metrics("File load", len(file_data), load_time, f"file: {weight_file}")
                
                # 产出所有张量（跳过元数据）
                tensor_count = 0
                for name, tensor in tensors.items():
                    if name != "__metadata__":
                        yield name, tensor
                        tensor_count += 1
                
                logger.info(f"Loaded {tensor_count} tensors from {weight_file}")
                
                # 如果还有文件需要预取，提交新的异步读取请求
                # 这样就实现了真正的流水线：在处理当前文件的同时，下一个文件已经在传输中
                if next_submit_index < len(weight_files):
                    next_weight_file = weight_files[next_submit_index]
                    file_size = file_sizes.get(next_weight_file, 1024 * 1024 * 100)
                    next_server_info = file_server_info.get(next_weight_file, {"segment_name": server_name})
                    offset = next_server_info.get("offset", 0)
                    
                    # 直接从数组中获取缓冲区地址
                    buffer_index = next_submit_index % pipeline_window_size
                    buffer_ptr = buffer_manager.buffer_ptrs[buffer_index]
                    
                    # 提交异步读取请求（使用优化版本）
                    try:
                        batch_id, returned_buffer_addr, length = _read_weight_range_async_optimized(
                            rdma_client, next_server_info, next_weight_file, offset, file_size,
                            buffer_ptr
                        )
                        
                        # 将新任务添加到队列末尾
                        active_tasks.append(AsyncReadTaskOptimized(
                            weight_file=next_weight_file,
                            batch_id=batch_id,
                            buffer_addr=returned_buffer_addr,
                            buffer_index=buffer_index,  # 存储缓冲区索引而不是地址
                            length=length,
                            server_info=next_server_info
                        ))
                    except Exception as e:
                        logger.error(f"Failed to submit async read request for {next_weight_file}: {e}")
                        raise
                    
                    next_submit_index += 1
                    
            except Exception as e:
                logger.error(f"Failed to load weights from {weight_file}: {e}")
                raise
    finally:
        # 清理缓冲区管理器
        buffer_manager.cleanup()
    
    # 计算总体带宽
    _log_performance_metrics("Overall async loading transfer", total_bytes, total_transfer_time)
    _log_performance_metrics("Overall async loading load", total_bytes, total_load_time)


def _read_weight_range_async_optimized(rdma_client, server_info, file_path, offset, length, 
                                     buffer_ptr):
    """优化的异步读取函数，使用预注册内存"""
    server_name = server_info.get("segment_name", "localhost")
    remote_buffer_addr = offset
    
    # 使用预注册内存的准确地址
    local_buffer_addr = buffer_ptr
    
    # 提交异步读取请求
    batch_id = rdma_client.batch_transfer_async_read(
        server_name,
        [local_buffer_addr],
        [remote_buffer_addr],
        [length]    
    )
    
    if batch_id <= 0:
        raise RuntimeError(f"Failed to submit async read request for {file_path}")
    
    return batch_id, local_buffer_addr, length  # 返回实际使用的缓冲区地址


def _read_weight_range_optimized(
    rdma_client: Any,
    server_info: Dict[str, Any],
    file_path: str,
    offset: int,
    length: int,
    buffer: Any,
    buffer_ptr: int
) -> bytes:
    """
    Args:
        rdma_client: Mooncake Transfer Engine client
        server_info: Server information
        file_path: Path to the file
        offset: Offset in the file to start reading
        length: Number of bytes to read
        buffer: Pre-allocated numpy buffer
        buffer_ptr: Pointer to the pre-allocated buffer
        
    Returns:
        The data read from the file as bytes
    """
    # Validate inputs
    if length <= 0:
        logger.warning(f"Requested to read {length} bytes from {file_path}, returning empty data")
        return b""
    
    server_name = server_info.get("segment_name", "localhost")
    logger.debug(f"Reading {length} bytes from {file_path} at offset {offset} on server {server_name} (optimized version)")
    
    try:
        # Use the offset as the remote buffer address
        remote_buffer_addr = offset
        
        # Perform RDMA read using mooncake engine API directly into pre-registered buffer
        ret = rdma_client.transfer_sync_read(
            server_name,
            buffer_ptr,
            remote_buffer_addr,
            length
        )
        
        if ret != 0:
            raise RuntimeError(f"RDMA read failed for {file_path} offset {offset} length {length} with code {ret}")
        
        data = buffer[:length].tobytes()
        logger.debug(f"Successfully read {len(data)} bytes from {file_path} using pre-registered buffer")
        return data
    except Exception as e:
        logger.error(f"Failed to read weight range from {file_path} on server {server_name}: {e}")
        raise


def _connect_to_etcd(etcd_host: str, etcd_port: int):
    """连接到ETCD服务器
    
    Args:
        etcd_host: ETCD服务器主机名
        etcd_port: ETCD服务器端口
        
    Returns:
        etcd3.Etcd3Client: ETCD客户端实例
    """
    logger.info(f"Connecting to ETCD server {etcd_host}:{etcd_port}...")
    try:
        etcd = etcd3.client(host=etcd_host, port=etcd_port)
        logger.info("ETCD connection successful!")
        return etcd
    except Exception as e:
        logger.error(f"ETCD connection failed: {e}")
        return None

def _get_model_weight_files(etcd_client, model_name: str) -> List[str]:
    """从ETCD获取模型的所有权重文件
    
    Args:
        etcd_client: ETCD客户端实例
        model_name: 模型名称
        
    Returns:
        权重文件列表
    """
    logger.info(f"Querying weight files for model '{model_name}'...")
    
    try:
        prefix = f"mooncake/checkpoint/{model_name}"
        weight_files = []
        
        # 获取所有以模型名称为前缀的键
        for value, metadata in etcd_client.get_prefix(prefix):
            if metadata and hasattr(metadata, 'key'):
                key = metadata.key.decode('utf-8')
                # 只获取权重文件（以.safetensors结尾的文件）
                if key.endswith('.safetensors'):
                    weight_files.append(key)
        
        logger.info(f"Found {len(weight_files)} weight files")
        return weight_files
    except Exception as e:
        logger.error(f"Error querying weight files: {e}")
        return []


def _get_weight_file_metadata(etcd_client, weight_file_key: str) -> Optional[Dict]:
    """从ETCD获取权重文件的元数据
    
    Args:
        etcd_client: ETCD客户端实例
        weight_file_key: 权重文件的键
        
    Returns:
        权重文件元数据，如果未找到则返回None
    """
    logger.debug(f"Querying metadata for weight file, key: {weight_file_key}")
    
    try:
        value, metadata = etcd_client.get(weight_file_key)
        
        if value:
            try:
                payload = json.loads(value.decode('utf-8'))
                return payload
            except json.JSONDecodeError as e:
                logger.error(f"  JSON decode failed: {e}")
                return None
        else:
            logger.warning(f"Metadata not found for key '{weight_file_key}'")
            return None
    except Exception as e:
        logger.error(f"Error querying weight file metadata: {e}")
        return None

def _extract_server_info_from_metadata(metadata: Dict) -> Optional[Dict]:
    """从元数据中提取服务器信息和偏移量
    
    Args:
        metadata: 权重文件元数据
        
    Returns:
        包含服务器信息和偏移量的字典，如果未找到则返回None
    """
    if not metadata:
        return None
        
    server_info = {}
    
    # 检查 shards 字段
    if "shards" in metadata and metadata["shards"]:
        # 获取第一个分片的信息
        first_shard = metadata["shards"][0]
        
        # 直接使用Replica副本信息，避免复杂的Gold优先逻辑
        if "replica_list" in first_shard and first_shard["replica_list"]:
            replica_info = first_shard["replica_list"][0]
            if "segment_name" in replica_info:
                server_info["segment_name"] = replica_info["segment_name"]
            if "offset" in replica_info:
                server_info["offset"] = replica_info["offset"]
        # 如果没有Replica副本，才检查Gold副本
        elif "gold" in first_shard and first_shard["gold"]:
            gold_info = first_shard["gold"][0]
            if "segment_name" in gold_info:
                server_info["segment_name"] = gold_info["segment_name"]
            if "offset" in gold_info:
                server_info["offset"] = gold_info["offset"]
    
    # 检查顶层字段
    if "segment_name" in metadata:
        server_info["segment_name"] = metadata["segment_name"]
    if "offset" in metadata:
        server_info["offset"] = metadata["offset"]
    
    return server_info if server_info else None

def _extract_servers_from_weight_files_with_cache(etcd_client, weight_files: List[str], metadata_cache: Dict[str, Dict]) -> List[str]:
    """从权重文件的元数据中提取服务器信息（使用缓存版本）
    
    Args:
        etcd_client: ETCD客户端实例
        weight_files: 权重文件列表
        metadata_cache: 元数据缓存
        
    Returns:
        服务器信息列表
    """
    servers = set()
    
    logger.info("Extracting server information from weight file metadata (cached version)...")
    
    # 遍历前几个权重文件以提取服务器信息（目前假定所有文件的服务器信息是相同的）
    for i, weight_file_key in enumerate(weight_files[:3]):  # 只检查前3个文件
        logger.debug(f"  Checking weight file: {weight_file_key}")
        
        # 优先使用缓存的元数据
        metadata = metadata_cache.get(weight_file_key)
        if not metadata:
            # 如果缓存中没有，再从ETCD获取
            metadata = _get_weight_file_metadata(etcd_client, weight_file_key)
        
        if metadata:
            # 尝试从不同的字段中提取服务器信息
            # 检查shards字段
            if "shards" in metadata:
                for shard_idx, shard in enumerate(metadata["shards"]):
                    logger.debug(f"    Processing shard {shard_idx}...")
                    # 检查Gold副本
                    gold = shard.get("gold", [])
                    for loc_idx, loc in enumerate(gold):
                        segment_name = loc.get("segment_name")
                        if segment_name:
                            servers.add(segment_name)
                            logger.debug(f"      Found Gold server: {segment_name}")
                        else:
                            logger.debug(f"      Gold location {loc_idx} missing segment_name field: {loc}")
                    
                    # 检查Replica副本
                    replicas = shard.get("replica_list", [])
                    for loc_idx, loc in enumerate(replicas):
                        segment_name = loc.get("segment_name")
                        if segment_name:
                            servers.add(segment_name)
                            logger.debug(f"      Found Replica server: {segment_name}")
                        else:
                            logger.debug(f"      Replica location {loc_idx} missing segment_name field: {loc}")
            
            # 检查顶层的gold和replica_list字段
            gold_list = metadata.get("gold", []) + metadata.get("Gold", [])
            for loc_idx, loc in enumerate(gold_list):
                segment_name = loc.get("segment_name") or loc.get("SegmentName")
                if segment_name:
                    servers.add(segment_name)
                    logger.debug(f"    Found top-level Gold server: {segment_name}")
                else:
                    logger.debug(f"    Top-level Gold location {loc_idx} missing segment_name field: {loc}")
                    
            replica_list = metadata.get("replica_list", []) + metadata.get("ReplicaList", [])
            for loc_idx, loc in enumerate(replica_list):
                segment_name = loc.get("segment_name") or loc.get("SegmentName")
                if segment_name:
                    servers.add(segment_name)
                    logger.debug(f"    Found top-level Replica server: {segment_name}")
                else:
                    logger.debug(f"    Top-level Replica location {loc_idx} missing segment_name field: {loc}")
        
        # 如果已经找到了服务器，可以提前结束
        if servers:
            logger.debug(f"  Found {len(servers)} servers, ending search early")
            break
    
    server_list = list(servers)
    logger.info(f"Discovered {len(server_list)} unique servers: {server_list}")
    return server_list


def _load_weights_batch(
    rdma_client: Any,
    weight_files: List[str],
    file_sizes: Dict[str, int],
    file_server_info: Dict[str, Dict],
    server_name: str
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """批量加载权重文件（在初始化时并行下载所有文件）
    
    Yields:
        Tuples of (weight_name, weight_tensor)
    """
    import time
    from collections import defaultdict
    
    logger.info("Starting batch weight loading (parallel downloading all files)")
    
    # 记录初始化缓冲区时间
    init_start_time = time.perf_counter()
    
    # 获取最大文件大小
    max_file_size = max(file_sizes.values()) if file_sizes else 1024 * 1024 * 100
    num_files = len(weight_files)
    
    logger.info(f"Max file size: {max_file_size / (1024*1024*1024):.2f} GB, Number of files: {num_files}")
    
    # 初始化缓冲区管理器，为每个文件创建一个缓冲区
    buffer_manager = RDMABufferManager()
    try:
        buffer_manager.initialize(rdma_client, max_file_size, num_files)
        buffers = buffer_manager.buffers
        buffer_ptrs = buffer_manager.buffer_ptrs
    except Exception as e:
        logger.error(f"Failed to initialize buffer manager for batch loading: {e}")
        raise RuntimeError(f"Failed to initialize buffer manager for batch loading: {e}")
    
    init_end_time = time.perf_counter()
    init_time = init_end_time - init_start_time
    _log_performance_metrics("Buffer initialization", 0, init_time)
    
    try:
        # 按服务器分组文件
        server_file_groups = defaultdict(list)
        total_bytes = 0
        
        for i, weight_file in enumerate(weight_files):
            # 从缓存中获取文件大小和服务器信息
            file_size = file_sizes.get(weight_file, 1024 * 1024 * 100)  # 默认100MB
            server_info = file_server_info.get(weight_file, {"segment_name": server_name})
            server_name_actual = server_info.get("segment_name", server_name)
            
            server_file_groups[server_name_actual].append({
                "index": i,
                "weight_file": weight_file,
                "file_size": file_size,
                "offset": server_info.get("offset", 0)
            })
            total_bytes += file_size
        
        # 为每个服务器提交批量读取请求
        batch_ids = []
        
        transfer_start_time = time.perf_counter()
        
        for server_name_actual, file_group in server_file_groups.items():
            # 为当前服务器准备批量请求
            local_buffer_addrs = []
            remote_buffer_addrs = []
            lengths = []
            
            for file_info in file_group:
                index = file_info["index"]
                file_size = file_info["file_size"]
                offset = file_info["offset"]
                
                # 获取对应缓冲区的地址
                buffer_ptr = buffer_ptrs[index]
                
                # 添加到批量请求列表
                local_buffer_addrs.append(buffer_ptr)
                remote_buffer_addrs.append(offset)
                lengths.append(file_size)
            
            # 使用批量异步读取接口提交当前服务器的请求
            logger.info(f"Submitting batch read request for {len(file_group)} files from server {server_name_actual}")
            
            batch_id = rdma_client.batch_transfer_async_read(
                server_name_actual,
                local_buffer_addrs,
                remote_buffer_addrs,
                lengths
            )
            
            if batch_id <= 0:
                raise RuntimeError(f"Failed to submit batch read request for server {server_name_actual}")
            
            batch_ids.append(batch_id)
        
        # 等待所有文件传输完成
        logger.info("Waiting for all files to be downloaded...")
        for batch_id in batch_ids:
            while True:
                status = rdma_client.get_batch_transfer_status([batch_id])
                if status == 0:  # 传输成功完成
                    break
                elif status < 0:  # 传输失败
                    raise RuntimeError(f"Batch read failed with status {status}")
                time.sleep(0.001)  # 1ms
        
        transfer_end_time = time.perf_counter()
        transfer_time = transfer_end_time - transfer_start_time
        
        _log_performance_metrics("Batch download", total_bytes, transfer_time)
        
        # 处理所有下载的文件数据
        total_load_time = 0.0
        load_total_bytes = 0
        
        for weight_file in weight_files:
            # 找到对应的缓冲区索引
            file_index = weight_files.index(weight_file)
            file_size = file_sizes.get(weight_file, 1024 * 1024 * 100)  # 默认100MB
            
            logger.info(f"Loading tensors from {weight_file}")
            
            try:
                copy_start_time = time.perf_counter()
                buffer = buffers[file_index]
                file_data = buffer[:file_size].tobytes()
                copy_end_time = time.perf_counter()
                _log_performance_metrics("File copy", len(file_data), copy_end_time - copy_start_time, f"file: {weight_file}")

                # 记录safetensors.load()时间
                load_start_time = time.perf_counter()
                
                tensors = load(file_data)
                
                load_end_time = time.perf_counter()
                load_time = load_end_time - load_start_time
                total_load_time += load_time
                load_total_bytes += len(file_data)
                
                _log_performance_metrics("File load", len(file_data), load_time, f"file: {weight_file}")
                
                tensor_count = 0
                for name, tensor in tensors.items():
                    if name != "__metadata__":
                        yield name, tensor
                        tensor_count += 1
                
                logger.info(f"Loaded {tensor_count} tensors from {weight_file}")
            except Exception as e:
                logger.error(f"Failed to load tensors from {weight_file}: {e}")
                raise
        
        # 记录总体加载时间
        _log_performance_metrics("Overall batch loading", load_total_bytes, total_load_time)
                
    finally:
        # 清理缓冲区管理器
        buffer_manager.cleanup()
        
    logger.info("Completed batch weight loading")


def _load_weights_async_safe_open(
    rdma_client: Any,
    weight_files: List[str],
    file_sizes: Dict[str, int],
    file_server_info: Dict[str, Dict],
    server_name: str,
    pipeline_window_size: int
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """使用共享内存和safe_open的异步加载权重文件实现
    
    Yields:
        Tuples of (weight_name, weight_tensor)
    """
    import time
    from collections import deque
    import os
    
    # 记录初始化缓冲区时间
    init_start_time = time.perf_counter()
    
    # 初始化共享内存缓冲区
    max_file_size = max(file_sizes.values()) if file_sizes else 1024 * 1024 * 100
    # 创建缓冲区管理器实例，使用共享内存
    buffer_manager = RDMABufferManager()
    try:
        buffer_manager.initialize(rdma_client, max_file_size, pipeline_window_size, use_shared_memory=True)
        buffer_ptrs = buffer_manager.buffer_ptrs
        shared_memory_names = buffer_manager.shared_memory_names
    except Exception as e:
        raise RuntimeError(f"Failed to initialize buffer manager for async safe_open loading: {e}")
    
    init_end_time = time.perf_counter()
    init_time = init_end_time - init_start_time
    _log_performance_metrics("Shared memory initialization", 0, init_time)
    
    # 使用双端队列存储正在进行的任务
    active_tasks = deque()
    
    try:
        # 提交初始窗口大小的异步读取请求
        for i in range(min(pipeline_window_size, len(weight_files))):
            weight_file = weight_files[i]
            file_size = file_sizes.get(weight_file, 1024 * 1024 * 100)
            server_info = file_server_info.get(weight_file, {"segment_name": server_name})
            offset = server_info.get("offset", 0)
            
            # 直接从数组中获取缓冲区地址
            buffer_ptr = buffer_ptrs[i]
            
            # 提交异步读取请求，直接读取到共享内存
            try:
                batch_id, returned_buffer_addr, length = _read_weight_range_async_optimized(
                    rdma_client, server_info, weight_file, offset, file_size,
                    buffer_ptr
                )
                
                active_tasks.append(AsyncReadTaskOptimized(
                    weight_file=weight_file,
                    batch_id=batch_id,
                    buffer_addr=returned_buffer_addr,
                    buffer_index=i,  # 存储缓冲区索引
                    length=length,
                    server_info=server_info
                ))
            except Exception as e:
                logger.error(f"Failed to submit async read request for {weight_file}: {e}")
                raise

        # 处理所有权重文件
        next_submit_index = pipeline_window_size
        total_bytes = 0
        total_transfer_time = 0.0
        total_load_time = 0.0
        
        # 继续处理直到所有任务完成
        while active_tasks:
            # 获取最早提交的任务
            current_task = active_tasks.popleft()
            weight_file = current_task.weight_file
            batch_id = current_task.batch_id
            buffer_index = current_task.buffer_index
            length = current_task.length
            server_info = current_task.server_info
            
            logger.info(f"Processing weights from {weight_file} via RDMA with safe_open")
            
            try:
                # 等待当前文件传输完成
                _wait_for_async_read_completion_optimized(rdma_client, batch_id)
                
                # 记录safe_open时间
                load_start_time = time.perf_counter()
                
                # 直接使用共享内存文件路径，避免内存拷贝
                shm_file_path = f"/dev/shm/{shared_memory_names[buffer_index]}"
                
                # 检查共享内存文件是否存在以及大小是否正确
                if not os.path.exists(shm_file_path):
                    raise RuntimeError(f"Shared memory file {shm_file_path} does not exist")
                
                # 获取文件大小
                actual_file_size = os.path.getsize(shm_file_path)
                logger.info(f"Shared memory file {shm_file_path} size: {actual_file_size} bytes, expected: {length} bytes")

                # safe_open需要文件大小与safetensors元数据中的大小完全匹配。
                # 我们的共享内存缓冲区是按最大文件大小预分配的，并且在流水线中重用。
                # 这意味着支持文件的大小可能与上一个使用者的大小不匹配（可能更大或更小）。
                # 因此，我们必须在每次使用前将文件调整为确切的预期大小。
                if actual_file_size != length:
                    logger.info(f"Resizing shared memory file {shm_file_path} from {actual_file_size} to {length} bytes")
                    try:
                        os.truncate(shm_file_path, length)
                        actual_file_size = length  # 更新文件大小变量
                    except OSError as e:
                        logger.error(f"Failed to resize shared memory file {shm_file_path}: {e}")
                        raise
                
                # 使用safe_open直接从共享内存文件路径加载张量
                tensor_count = 0
                try:
                    with safe_open(shm_file_path, framework="pt", device="cpu") as f:
                        for name in f.keys():
                            if name != "__metadata__":
                                tensor = f.get_tensor(name)
                                yield name, tensor
                                tensor_count += 1
                except Exception as e:
                    logger.error(f"Failed to load tensors from {shm_file_path} with safe_open: {e}")
                    logger.error(f"File size: {actual_file_size}, Expected size: {length}")
                    # 检查文件头几个字节以帮助诊断问题
                    try:
                        with open(shm_file_path, 'rb') as debug_f:
                            header_data = debug_f.read(1024)  # 读取前1024字节
                            logger.error(f"First 1024 bytes of file (hex): {header_data.hex()}")
                    except Exception as debug_e:
                        logger.error(f"Failed to read file for debugging: {debug_e}")
                    raise
                
                load_end_time = time.perf_counter()
                load_time = load_end_time - load_start_time
                total_load_time += load_time
                
                _log_performance_metrics("File load", length, load_time, f"file: {weight_file}")
                logger.info(f"Loaded {tensor_count} tensors from {weight_file} using safe_open with shared memory")
                
                # 如果还有文件需要预取，提交新的异步读取请求
                if next_submit_index < len(weight_files):
                    next_weight_file = weight_files[next_submit_index]
                    file_size = file_sizes.get(next_weight_file, 1024 * 1024 * 100)
                    next_server_info = file_server_info.get(next_weight_file, {"segment_name": server_name})
                    offset = next_server_info.get("offset", 0)
                    
                    buffer_index = next_submit_index % pipeline_window_size
                    buffer_ptr = buffer_ptrs[buffer_index]
                    
                    # 提交异步读取请求
                    try:
                        batch_id, returned_buffer_addr, actual_length = _read_weight_range_async_optimized(
                            rdma_client, next_server_info, next_weight_file, offset, file_size,
                            buffer_ptr
                        )
                        
                        # 将新任务添加到队列末尾
                        active_tasks.append(AsyncReadTaskOptimized(
                            weight_file=next_weight_file,
                            batch_id=batch_id,
                            buffer_addr=returned_buffer_addr,
                            buffer_index=buffer_index,
                            length=actual_length,  # 使用实际长度
                            server_info=next_server_info
                        ))
                    except Exception as e:
                        logger.error(f"Failed to submit async read request for {next_weight_file}: {e}")
                        raise
                    
                    next_submit_index += 1
                    
            except Exception as e:
                logger.error(f"Failed to load weights from {weight_file}: {e}")
                raise
                
    finally:
        # 清理缓冲区管理器
        buffer_manager.cleanup()
    
    # 计算总体带宽
    _log_performance_metrics("Overall async loading transfer", total_bytes, total_transfer_time)
    _log_performance_metrics("Overall async loading load", total_bytes, total_load_time)

