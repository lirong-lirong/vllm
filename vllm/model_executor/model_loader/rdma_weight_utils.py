# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for loading model weights via RDMA directly from remote storage."""
import json
from collections.abc import Generator
from typing import Dict, Any, Tuple, List, Optional

import torch
from safetensors.torch import load

from vllm.logger import init_logger

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
    
    logger.info(f"Starting RDMA weight loading for model: {model_name}")
    
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
    
    # 4. 从权重文件中提取服务器信息
    servers = _extract_servers_from_weight_files_with_cache(etcd_client, weight_files, file_metadata_cache)
    if not servers:
        raise RuntimeError("No servers found for weight files")
    
    logger.info(f"Discovered {len(servers)} servers: {servers}")
    
    # 5. 使用第一个服务器进行连接测试和权重加载
    server_name = servers[0]
    
    logger.info(f"Using server {server_name} for RDMA weight loading")
    
    for weight_file in weight_files:
        logger.info(f"Loading weights from {weight_file} via RDMA")
        
        try:
            # 从缓存中获取文件大小和服务器信息
            file_size = file_sizes.get(weight_file, 1024 * 1024 * 100)  # 默认100MB
            server_info = file_server_info.get(weight_file, {"segment_name": server_name})
            offset = server_info.get("offset", 0)
            
            # 通过RDMA一次性读取整个文件
            file_data = _read_weight_range_optimized(rdma_client, server_info, weight_file, offset, file_size)
            
            # 使用safetensors.load()从内存数据加载所有张量
            tensors = load(file_data)
            
            # 产出所有张量（跳过元数据）
            tensor_count = 0
            for name, tensor in tensors.items():
                if name != "__metadata__":
                    yield name, tensor
                    tensor_count += 1
            
            logger.info(f"Loaded {tensor_count} tensors from {weight_file}")
        except Exception as e:
            logger.error(f"Failed to load weights from {weight_file}: {e}")
            raise
    
    logger.info(f"Completed RDMA weight loading for model: {model_name}")


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
        # 构造前缀
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


def _extract_servers_from_weight_files(etcd_client, weight_files: List[str]) -> List[str]:
    """从权重文件的元数据中提取服务器信息
    
    Args:
        etcd_client: ETCD客户端实例
        weight_files: 权重文件列表
        
    Returns:
        服务器信息列表
    """
    servers = set()  # 使用集合避免重复
    
    logger.info("Extracting server information from weight file metadata...")
    
    # 遍历前几个权重文件以提取服务器信息（通常所有文件的服务器信息是相同的）
    for i, weight_file_key in enumerate(weight_files[:3]):  # 只检查前3个文件
        logger.debug(f"  Checking weight file: {weight_file_key}")
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


def _extract_servers_from_weight_files_with_cache(etcd_client, weight_files: List[str], metadata_cache: Dict[str, Dict]) -> List[str]:
    """从权重文件的元数据中提取服务器信息（使用缓存版本）
    
    Args:
        etcd_client: ETCD客户端实例
        weight_files: 权重文件列表
        metadata_cache: 元数据缓存
        
    Returns:
        服务器信息列表
    """
    servers = set()  # 使用集合避免重复
    
    logger.info("Extracting server information from weight file metadata (cached version)...")
    
    # 遍历前几个权重文件以提取服务器信息（通常所有文件的服务器信息是相同的）
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


def _get_file_size_from_etcd(etcd_client, weight_file_key: str) -> int:
    """从ETCD获取文件大小
    
    Args:
        etcd_client: ETCD客户端实例
        weight_file_key: 权重文件的键
        
    Returns:
        文件大小（字节）
    """
    logger.debug(f"Getting file size from ETCD for {weight_file_key}")
    
    try:
        metadata = _get_weight_file_metadata(etcd_client, weight_file_key)
        if metadata:
            # 尝试从不同字段获取文件大小
            file_size = (
                metadata.get("size") or 
                metadata.get("Size") or 
                metadata.get("total_size") or 
                metadata.get("TotalSize") or
                1024 * 1024 * 100  # 默认100MB
            )
            logger.debug(f"File size for {weight_file_key}: {file_size} bytes")
            return file_size
        else:
            logger.warning(f"Could not get metadata for {weight_file_key}, using default size")
            return 1024 * 1024 * 100  # 默认100MB
    except Exception as e:
        logger.error(f"Error getting file size from ETCD: {e}")
        return 1024 * 1024 * 100  # 默认100MB


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


def _get_server_info_from_etcd(etcd_client, weight_file_key: str) -> Optional[Dict]:
    """从ETCD获取服务器信息和偏移量
    
    Args:
        etcd_client: ETCD客户端实例
        weight_file_key: 权重文件的键
        
    Returns:
        包含服务器信息和偏移量的字典，如果未找到则返回None
    """
    logger.debug(f"Getting server info from ETCD for {weight_file_key}")
    
    try:
        metadata = _get_weight_file_metadata(etcd_client, weight_file_key)
        if metadata:
            server_info = _extract_server_info_from_metadata(metadata)
            if server_info:
                logger.debug(f"Server info for {weight_file_key}: {server_info}")
                return server_info
            else:
                logger.warning(f"Could not extract server info from metadata for {weight_file_key}")
                return None
        else:
            logger.warning(f"Could not get metadata for {weight_file_key}")
            return None
    except Exception as e:
        logger.error(f"Error getting server info from ETCD: {e}")
        return None


# 全局预注册缓冲区（实验性优化）
_pre_registered_buffer = None
_pre_registered_buffer_size = 0
_pre_registered_buffer_ptr = 0


def _initialize_pre_registered_buffer(rdma_client: Any, size: int = 8 * 1024 * 1024 * 1024):
    """初始化预注册缓冲区（实验性优化）
    
    Args:
        rdma_client: Mooncake Transfer Engine client
        size: 缓冲区大小，默认8GB
    """
    global _pre_registered_buffer, _pre_registered_buffer_size, _pre_registered_buffer_ptr
    
    if _pre_registered_buffer is not None:
        return  # 已经初始化过了
    
    try:
        import numpy as np
        logger.info(f"Initializing pre-registered buffer of size {size} bytes ({size / (1024*1024*1024):.2f} GB)")
        
        # 预分配缓冲区
        _pre_registered_buffer = np.empty(size, dtype=np.uint8)
        _pre_registered_buffer_ptr = _pre_registered_buffer.ctypes.data
        _pre_registered_buffer_size = size
        
        # 注册内存
        ret = rdma_client.register_memory(_pre_registered_buffer_ptr, size)
        if ret != 0:
            logger.warning(f"Failed to register memory for pre-registered buffer, falling back to managed buffers. Return code: {ret}")
            _pre_registered_buffer = None
            _pre_registered_buffer_size = 0
            _pre_registered_buffer_ptr = 0
        else:
            logger.info("Successfully initialized pre-registered buffer")
    except Exception as e:
        logger.warning(f"Failed to initialize pre-registered buffer, falling back to managed buffers: {e}")
        _pre_registered_buffer = None
        _pre_registered_buffer_size = 0
        _pre_registered_buffer_ptr = 0


def _read_weight_range_optimized(
    rdma_client: Any,
    server_info: Dict[str, Any],
    file_path: str,
    offset: int,
    length: int
) -> bytes:
    """优化版本的读取远程文件数据范围函数，使用预注册缓冲区避免重复内存注册和二次拷贝
    
    Args:
        rdma_client: Mooncake Transfer Engine client
        server_info: Server information
        file_path: Path to the file
        offset: Offset in the file to start reading
        length: Number of bytes to read
        
    Returns:
        The data read from the file as bytes
    """
    # Validate inputs
    if length <= 0:
        logger.warning(f"Requested to read {length} bytes from {file_path}, returning empty data")
        return b""
    
    server_name = server_info.get("segment_name", "localhost")
    logger.debug(f"Reading {length} bytes from {file_path} at offset {offset} on server {server_name} (optimized version)")
    
    # 尝试初始化预注册缓冲区
    _initialize_pre_registered_buffer(rdma_client)
    
    # 如果预注册缓冲区可用且足够大，使用它
    global _pre_registered_buffer, _pre_registered_buffer_size, _pre_registered_buffer_ptr
    
    if _pre_registered_buffer is not None and _pre_registered_buffer_size >= length:
        try:
            logger.debug(f"Using pre-registered buffer of size {_pre_registered_buffer_size} for RDMA read")
            
            # Use the offset as the remote buffer address
            remote_buffer_addr = offset
            
            # Perform RDMA read using mooncake engine API directly into pre-registered buffer
            ret = rdma_client.transfer_sync_read(
                server_name,
                _pre_registered_buffer_ptr,
                remote_buffer_addr,
                length
            )
            
            if ret != 0:
                raise RuntimeError(f"RDMA read failed for {file_path} offset {offset} length {length} with code {ret}")
            
            # 直接从预注册缓冲区返回数据，避免二次拷贝
            # 注意：这里我们创建一个bytes对象的视图，而不是拷贝数据
            data = _pre_registered_buffer[:length].tobytes()
            logger.debug(f"Successfully read {len(data)} bytes from {file_path} using pre-registered buffer")
            return data
        except Exception as e:
            logger.warning(f"Failed to use pre-registered buffer, falling back to managed buffers: {e}")
    
    # 回退到原始的托管缓冲区方法
    try:
        # Allocate managed buffer
        buffer_addr = rdma_client.allocate_managed_buffer(length)
        if buffer_addr == 0:
            raise RuntimeError(f"Failed to allocate managed buffer of size {length}")
        
        logger.debug(f"Allocated buffer at address {buffer_addr}")
        
        try:
            # Use the offset as the remote buffer address
            remote_buffer_addr = offset
            
            # Perform RDMA read using mooncake engine API
            ret = rdma_client.transfer_sync_read(
                server_name,
                buffer_addr,
                remote_buffer_addr,
                length
            )
            
            if ret != 0:
                raise RuntimeError(f"RDMA read failed for {file_path} offset {offset} length {length} with code {ret}")
            
            # Read bytes from buffer
            data = rdma_client.read_bytes_from_buffer(buffer_addr, length)
            logger.debug(f"Successfully read {len(data)} bytes from {file_path}")
            return data
        finally:
            # Free the buffer
            rdma_client.free_managed_buffer(buffer_addr, length)
            logger.debug(f"Freed buffer at address {buffer_addr}")
    except Exception as e:
        logger.error(f"Failed to read weight range from {file_path} on server {server_name}: {e}")
        raise


def _read_weight_range(
    rdma_client: Any,
    server_info: Dict[str, Any],
    file_path: str,
    offset: int,
    length: int
) -> bytes:
    """Read a range of data from a remote file via RDMA.
    
    Args:
        rdma_client: Mooncake Transfer Engine client
        server_info: Server information
        file_path: Path to the file
        offset: Offset in the file to start reading
        length: Number of bytes to read
        
    Returns:
        The data read from the file as bytes
    """
    # 使用优化版本
    return _read_weight_range_optimized(rdma_client, server_info, file_path, offset, length)
