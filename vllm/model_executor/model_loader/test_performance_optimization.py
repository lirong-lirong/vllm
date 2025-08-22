#!/usr/bin/env python3
"""
性能优化测试脚本，用于验证RDMA模型加载器的优化效果
"""

import argparse
import os
import sys
import time

# 添加vllm到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    import torch
    from mooncake.engine import TransferEngine
    from vllm.model_executor.model_loader.rdma_weight_utils import (
        _connect_to_etcd,
        _get_model_weight_files,
        _get_file_size_from_etcd,
        _get_server_info_from_etcd,
        _read_weight_range_optimized
    )
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def test_etcd_access_optimization(etcd_host, etcd_port, model_name):
    """测试ETCD访问优化效果"""
    print("=== 测试ETCD访问优化 ===")
    
    # 连接到ETCD
    etcd_client = _connect_to_etcd(etcd_host, etcd_port)
    if not etcd_client:
        print("无法连接到ETCD服务器")
        return
    
    # 获取模型权重文件
    weight_files = _get_model_weight_files(etcd_client, model_name)
    if not weight_files:
        print(f"未找到模型 {model_name} 的权重文件")
        return
    
    print(f"找到 {len(weight_files)} 个权重文件")
    
    # 测试旧方法：逐个访问ETCD获取文件大小
    print("\n测试旧方法（逐个访问ETCD）...")
    start_time = time.time()
    old_file_sizes = {}
    old_server_info = {}
    
    for weight_file in weight_files:
        old_file_sizes[weight_file] = _get_file_size_from_etcd(etcd_client, weight_file)
        old_server_info[weight_file] = _get_server_info_from_etcd(etcd_client, weight_file)
    
    old_time = time.time() - start_time
    print(f"旧方法耗时: {old_time:.4f} 秒")
    
    # 测试新方法：批量获取元数据
    print("\n测试新方法（批量获取元数据）...")
    start_time = time.time()
    
    # 预先获取所有权重文件的元数据
    from vllm.model_executor.model_loader.rdma_weight_utils import _get_weight_file_metadata
    file_metadata_cache = {}
    new_file_sizes = {}
    new_server_info = {}
    
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
            new_file_sizes[weight_file] = file_size
            
            # 提取服务器信息和偏移量
            from vllm.model_executor.model_loader.rdma_weight_utils import _extract_server_info_from_metadata
            server_info = _extract_server_info_from_metadata(metadata)
            if server_info:
                new_server_info[weight_file] = server_info
    
    new_time = time.time() - start_time
    print(f"新方法耗时: {new_time:.4f} 秒")
    
    # 计算性能提升
    if old_time > 0:
        improvement = (old_time - new_time) / old_time * 100
        print(f"性能提升: {improvement:.2f}%")
    
    print(f"\n文件大小一致性检查: {old_file_sizes == new_file_sizes}")
    print(f"服务器信息一致性检查: {old_server_info == new_server_info}")


def test_memory_registration_optimization(etcd_host, etcd_port, model_name):
    """测试内存注册优化效果"""
    print("\n=== 测试内存注册优化 ===")
    
    # 连接到ETCD
    etcd_client = _connect_to_etcd(etcd_host, etcd_port)
    if not etcd_client:
        print("无法连接到ETCD服务器")
        return
    
    # 获取模型权重文件
    weight_files = _get_model_weight_files(etcd_client, model_name)
    if not weight_files:
        print(f"未找到模型 {model_name} 的权重文件")
        return
    
    # 获取第一个权重文件的信息
    first_weight_file = weight_files[0]
    file_size = _get_file_size_from_etcd(etcd_client, first_weight_file)
    server_info = _get_server_info_from_etcd(etcd_client, first_weight_file)
    
    if not server_info:
        print("无法获取服务器信息")
        return
    
    server_name = server_info.get("segment_name", "localhost")
    offset = server_info.get("offset", 0)
    
    # 初始化RDMA客户端
    rdma_client = TransferEngine()
    ret = rdma_client.initialize("localhost", f"{etcd_host}:{etcd_port}", "rdma", "")
    if ret != 0:
        print("无法初始化RDMA客户端")
        return
    
    try:
        # 测试旧方法：使用托管缓冲区
        print("\n测试旧方法（托管缓冲区）...")
        start_time = time.time()
        data_old = _read_weight_range_optimized.__code__ = None  # 临时禁用优化
        # 重新定义原始函数进行测试
        def _read_weight_range_original(rdma_client, server_info, file_path, offset, length):
            server_name = server_info.get("segment_name", "localhost")
            buffer_addr = rdma_client.allocate_managed_buffer(length)
            if buffer_addr == 0:
                raise RuntimeError(f"Failed to allocate managed buffer of size {length}")
            try:
                remote_buffer_addr = offset
                ret = rdma_client.transfer_sync_read(server_name, buffer_addr, remote_buffer_addr, length)
                if ret != 0:
                    raise RuntimeError(f"RDMA read failed for {file_path} offset {offset} length {length} with code {ret}")
                data = rdma_client.read_bytes_from_buffer(buffer_addr, length)
                return data
            finally:
                rdma_client.free_managed_buffer(buffer_addr, length)
        
        data_old = _read_weight_range_original(rdma_client, server_info, first_weight_file, offset, min(file_size, 1024))
        old_time = time.time() - start_time
        print(f"旧方法耗时: {old_time:.4f} 秒")
        print(f"读取数据大小: {len(data_old)} 字节")
        
        # 测试新方法：使用预注册缓冲区
        print("\n测试新方法（预注册缓冲区）...")
        start_time = time.time()
        # 重新启用优化函数
        data_new = _read_weight_range_optimized(rdma_client, server_info, first_weight_file, offset, min(file_size, 1024))
        new_time = time.time() - start_time
        print(f"新方法耗时: {new_time:.4f} 秒")
        print(f"读取数据大小: {len(data_new)} 字节")
        
        # 计算性能提升
        if old_time > 0 and new_time > 0:
            improvement = (old_time - new_time) / old_time * 100
            print(f"性能提升: {improvement:.2f}%")
        
        # 数据一致性检查
        print(f"数据一致性检查: {data_old == data_new}")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
    finally:
        # 清理RDMA客户端
        if hasattr(rdma_client, 'shutdown'):
            rdma_client.shutdown()


def main():
    parser = argparse.ArgumentParser(description="性能优化测试脚本")
    parser.add_argument("model_name", help="模型名称")
    parser.add_argument("--etcd-host", default="localhost", help="ETCD服务器主机名 (默认: localhost)")
    parser.add_argument("--etcd-port", type=int, default=2379, help="ETCD服务器端口 (默认: 2379)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RDMA模型加载器性能优化测试")
    print("=" * 60)
    print(f"模型名称: {args.model_name}")
    print(f"ETCD服务器: {args.etcd_host}:{args.etcd_port}")
    print("-" * 60)
    
    # 测试ETCD访问优化
    test_etcd_access_optimization(args.etcd_host, args.etcd_port, args.model_name)
    
    # 测试内存注册优化
    test_memory_registration_optimization(args.etcd_host, args.etcd_port, args.model_name)
    
    print("-" * 60)
    print("性能优化测试完成!")


if __name__ == "__main__":
    main()
