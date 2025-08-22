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
    
    # 3. 从权重文件中提取服务器信息
    servers = _extract_servers_from_weight_files(etcd_client, weight_files)
    if not servers:
        raise RuntimeError("No servers found for weight files")
    
    logger.info(f"Discovered {len(servers)} servers: {servers}")
    
    # 4. 使用第一个服务器进行连接测试和权重加载
    server_name = servers[0]
    server_info = {"segment_name": server_name}
    
    logger.info(f"Using server {server_name} for RDMA weight loading")
    
    for weight_file in weight_files:
        logger.info(f"Loading weights from {weight_file} via RDMA")
        
        try:
            # 1. 从ETCD获取文件大小
            file_size = _get_file_size_from_etcd(etcd_client, weight_file)
            
            # 2. 获取文件的服务器信息和偏移量
            server_info_with_offset = _get_server_info_from_etcd(etcd_client, weight_file)
            if server_info_with_offset:
                actual_server_name = server_info_with_offset.get("segment_name", server_name)
                offset = server_info_with_offset.get("offset", 0)
                server_info = {"segment_name": actual_server_name}
            else:
                offset = 0
            
            # 3. 通过RDMA一次性读取整个文件
            file_data = _read_weight_range(rdma_client, server_info, weight_file, offset, file_size)
            
            # 4. 使用safetensors.load()从内存数据加载所有张量
            tensors = load(file_data)
            
            # 5. 产出所有张量（跳过元数据）
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
            # 尝试从不同字段获取服务器信息
            server_info = {}
            
            # 检查 shards 字段
            if "shards" in metadata and metadata["shards"]:
                # 获取第一个分片的信息
                first_shard = metadata["shards"][0]
                
                # 检查Gold副本获取服务器地址和偏移量
                if "gold" in first_shard and first_shard["gold"]:
                    gold_info = first_shard["gold"][0]
                    if "segment_name" in gold_info:
                        server_info["segment_name"] = gold_info["segment_name"]
                    if "offset" in gold_info:
                        server_info["offset"] = gold_info["offset"]
                # 如果没有Gold副本，检查Replica副本
                elif "replica_list" in first_shard and first_shard["replica_list"]:
                    replica_info = first_shard["replica_list"][0]
                    if "segment_name" in replica_info:
                        server_info["segment_name"] = replica_info["segment_name"]
                    if "offset" in replica_info:
                        server_info["offset"] = replica_info["offset"]
            
            # 检查顶层字段
            if "segment_name" in metadata:
                server_info["segment_name"] = metadata["segment_name"]
            if "offset" in metadata:
                server_info["offset"] = metadata["offset"]
            
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
    # Validate inputs
    if length <= 0:
        logger.warning(f"Requested to read {length} bytes from {file_path}, returning empty data")
        return b""
    
    server_name = server_info.get("segment_name", "localhost")
    logger.debug(f"Reading {length} bytes from {file_path} at offset {offset} on server {server_name}")
    
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
