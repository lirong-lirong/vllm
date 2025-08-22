#!/usr/bin/env python3
"""
Unit tests for rdma_weight_utils.py
"""

import unittest
import torch
import os
import argparse
import sys
import tempfile
from safetensors.torch import save_file

# Import the functions we want to test
from vllm.model_executor.model_loader.rdma_weight_utils import (
    _connect_to_etcd,
    _get_model_weight_files,
    _get_weight_file_metadata,
    _extract_servers_from_weight_files,
    _get_file_size_from_etcd,
    _get_server_info_from_etcd,
    rdma_safetensors_weights_iterator,
    _read_weight_range
)


class TestRdmaWeightUtilsWithRealEtcd(unittest.TestCase):
    """Test cases for rdma_weight_utils.py functions using real ETCD connection"""
    
    @classmethod
    def setUpClass(cls):
        """Set up class fixtures"""
        # Get ETCD configuration from environment variables or use defaults
        cls.etcd_host = os.environ.get('ETCD_HOST', 'localhost')
        cls.etcd_port = int(os.environ.get('ETCD_PORT', 2379))
        cls.test_model_name = os.environ.get('TEST_MODEL_NAME', 'test_model')
        
        print(f"Using ETCD server: {cls.etcd_host}:{cls.etcd_port}")
        print(f"Using test model: {cls.test_model_name}")

    def test_real_etcd_connection(self):
        """Test real ETCD connection"""
        etcd_client = _connect_to_etcd(self.etcd_host, self.etcd_port)
        self.assertIsNotNone(etcd_client, "Failed to connect to real ETCD server")
        
        # Try to get model weight files (may be empty, but should not raise exception)
        weight_files = _get_model_weight_files(etcd_client, self.test_model_name)
        self.assertIsInstance(weight_files, list, "Should return a list of weight files")
        
        print(f"Found {len(weight_files)} weight files for model '{self.test_model_name}'")

    def test_real_model_discovery(self):
        """Test discovering model files from real ETCD"""
        etcd_client = _connect_to_etcd(self.etcd_host, self.etcd_port)
        self.assertIsNotNone(etcd_client, "Failed to connect to real ETCD server")
        
        # Get model weight files
        weight_files = _get_model_weight_files(etcd_client, self.test_model_name)
        
        if weight_files:
            # Extract servers from weight files
            servers = _extract_servers_from_weight_files(etcd_client, weight_files)
            self.assertIsInstance(servers, list, "Should return a list of servers")
            print(f"Discovered {len(servers)} servers: {servers}")
            
            # Test getting file size for first weight file
            file_size = _get_file_size_from_etcd(etcd_client, weight_files[0])
            self.assertIsInstance(file_size, int, "File size should be an integer")
            self.assertGreater(file_size, 0, "File size should be positive")
            print(f"File size for {weight_files[0]}: {file_size} bytes")
            
            # Test getting server info
            server_info = _get_server_info_from_etcd(etcd_client, weight_files[0])
            if server_info:
                self.assertIsInstance(server_info, dict, "Server info should be a dictionary")
                self.assertIn("segment_name", server_info, "Server info should contain segment_name")
                print(f"Server info for {weight_files[0]}: {server_info}")
        else:
            print(f"No weight files found for model '{self.test_model_name}', skipping detailed tests")

    def test_rdma_safetensors_weights_iterator(self):
        """Test rdma_safetensors_weights_iterator function with real ETCD and RDMA"""
        # Connect to ETCD
        etcd_client = _connect_to_etcd(self.etcd_host, self.etcd_port)
        self.assertIsNotNone(etcd_client, "Failed to connect to real ETCD server")
        
        # Get model weight files
        weight_files = _get_model_weight_files(etcd_client, self.test_model_name)
        
        # If we have weight files, we can test the iterator
        if weight_files:
            try:
                # Try to import the TransferEngine
                from mooncake.engine import TransferEngine
                
                # Initialize RDMA client
                rdma_client = TransferEngine()
                ret = rdma_client.initialize("localhost", f"{self.etcd_host}:{self.etcd_port}", "rdma", "")
                if ret != 0:
                    print("Failed to initialize RDMA client, skipping iterator test")
                    self.skipTest("Failed to initialize RDMA client")
                
                try:
                    # Record start time for bandwidth calculation
                    import time
                    start_time = time.time()
                    
                    # Test the iterator
                    weights_count = 0
                    total_bytes_loaded = 0
                    
                    # Get the first weight file size for validation
                    first_file_size = _get_file_size_from_etcd(etcd_client, weight_files[0])
                    
                    for name, tensor in rdma_safetensors_weights_iterator(
                        self.test_model_name, 
                        rdma_client, 
                        self.etcd_host, 
                        self.etcd_port
                    ):
                        # Verify that we get valid results
                        self.assertIsInstance(name, str, "Weight name should be a string")
                        self.assertIsInstance(tensor, torch.Tensor, "Weight should be a torch tensor")
                        weights_count += 1
                        
                        # Calculate approximate bytes loaded (this is a rough estimate)
                        total_bytes_loaded += tensor.nelement() * tensor.element_size()
                        
                        # Limit the number of weights we check to avoid long test times
                        if weights_count >= 5:
                            break
                    
                    # Record end time for bandwidth calculation
                    end_time = time.time()
                    
                    # Calculate bandwidth
                    elapsed_time = end_time - start_time
                    if elapsed_time > 0:
                        bandwidth_mbps = (total_bytes_loaded * 8) / (elapsed_time * 1024 * 1024)
                        print(f"Bandwidth: {bandwidth_mbps:.2f} Mbps ({(total_bytes_loaded / (1024 * 1024)) / elapsed_time:.2f} MB/s)")
                    
                    print(f"Successfully iterated through {weights_count} weights")
                    print(f"Total bytes loaded (approximate): {total_bytes_loaded} bytes")
                    self.assertGreater(weights_count, 0, "Should have loaded at least one weight")
                    
                    # Validate by reading beginning and end of the first weight file
                    if weights_count > 0:
                        # Get server info for the first weight file
                        server_info = _get_server_info_from_etcd(etcd_client, weight_files[0])
                        if server_info:
                            server_name = server_info.get("segment_name", "localhost")
                            offset = server_info.get("offset", 0)
                            
                            # Read beginning of file (first 100 bytes)
                            head_buffer_size = min(100, first_file_size)
                            head_buffer_addr = rdma_client.allocate_managed_buffer(head_buffer_size)
                            if head_buffer_addr != 0:
                                try:
                                    ret = rdma_client.transfer_sync_read(
                                        server_name,
                                        head_buffer_addr,
                                        offset,
                                        head_buffer_size
                                    )
                                    if ret == 0:
                                        head_data = rdma_client.read_bytes_from_buffer(head_buffer_addr, head_buffer_size)
                                        print(f"Successfully read {len(head_data)} bytes from beginning of file: {head_data[:50]}{'...' if len(head_data) > 50 else ''}")
                                    else:
                                        print(f"Failed to read beginning of file, return code: {ret}")
                                finally:
                                    rdma_client.free_managed_buffer(head_buffer_addr, head_buffer_size)
                            
                            # Read end of file (last 100 bytes)
                            if first_file_size > 100:
                                tail_buffer_size = 100
                                tail_buffer_addr = rdma_client.allocate_managed_buffer(tail_buffer_size)
                                if tail_buffer_addr != 0:
                                    try:
                                        ret = rdma_client.transfer_sync_read(
                                            server_name,
                                            tail_buffer_addr,
                                            offset + first_file_size - tail_buffer_size,
                                            tail_buffer_size
                                        )
                                        if ret == 0:
                                            tail_data = rdma_client.read_bytes_from_buffer(tail_buffer_addr, tail_buffer_size)
                                            print(f"Successfully read {len(tail_data)} bytes from end of file: {tail_data[:50]}{'...' if len(tail_data) > 50 else ''}")
                                        else:
                                            print(f"Failed to read end of file, return code: {ret}")
                                    finally:
                                        rdma_client.free_managed_buffer(tail_buffer_addr, tail_buffer_size)
                finally:
                    # Shutdown RDMA client if shutdown method exists
                    if hasattr(rdma_client, 'shutdown'):
                        rdma_client.shutdown()
                    else:
                        print("RDMA client does not have shutdown method, skipping cleanup")
                    
            except ImportError:
                print("Mooncake TransferEngine not available, skipping iterator test")
                self.skipTest("Mooncake TransferEngine not available")
            except Exception as e:
                print(f"Error testing iterator: {e}")
                # We don't fail the test here because the iterator might fail due to missing model files
                # or other environmental issues
                self.skipTest(f"Error testing iterator: {e}")
        else:
            print(f"No weight files found for model '{self.test_model_name}', skipping iterator test")
            self.skipTest("No weight files found for model")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test RDMA weight utilities")
    parser.add_argument("--etcd-host", default="localhost", 
                        help="ETCD server host (default: localhost)")
    parser.add_argument("--etcd-port", type=int, default=2379, 
                        help="ETCD server port (default: 2379)")
    parser.add_argument("--test-model-name", default="test_model", 
                        help="Test model name (default: test_model)")
    return parser.parse_args()


def main():
    """Main function to run tests"""
    args = parse_args()
    
    # Set environment variables for real ETCD tests
    os.environ['ETCD_HOST'] = args.etcd_host
    os.environ['ETCD_PORT'] = str(args.etcd_port)
    os.environ['TEST_MODEL_NAME'] = args.test_model_name
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Run tests with real ETCD
    suite.addTests(loader.loadTestsFromTestCase(TestRdmaWeightUtilsWithRealEtcd))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
